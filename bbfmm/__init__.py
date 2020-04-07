import scipy.signal
import aljpy
import numpy as np
import numba
from . import test, chebyshev, tree

KERNEL = test.quad_kernel
EPS = 1e-2

def limits(prob):
    points = np.concatenate([prob.sources, prob.targets])
    return np.stack([points.min(0) - EPS, points.max(0) + EPS])

def scale(prob):
    lims = limits(prob)
    lower, scale = lims[0], lims[1] - lims[0]
    return aljpy.dotdict(
        limits=lims,
        scale=scale,
        sources=(prob.sources - lower)/scale,
        charges=prob.charges,
        targets=(prob.targets - lower)/scale)

def leaf_sum(l, x, d):
    x = np.asarray(x)
    D = l.shape[1]
    totals = np.zeros((2**d,)*D + x.shape[1:])
    np.add.at(totals, tuple(l.T), x)
    return totals

def leaf_centers(d):
    return (1/2 + np.arange(2**d))/2**d

def tree_leaves(scaled, cutoff=5):
    D = scaled.sources.shape[1]
    sl = np.zeros((len(scaled.sources), D), dtype=int)
    tl = np.zeros((len(scaled.targets), D), dtype=int)

    #TODO: You can probably get very smart about this and just convert the xs
    # to ints and look at their binary representation.
    d = 0
    while True:
        s_done = leaf_sum(sl, 1, d).max() <= cutoff
        t_done = leaf_sum(tl, 1, d).max() <= cutoff
        if s_done and t_done:
            break

        centers = leaf_centers(d)
        sl = 2*sl + (scaled.sources >= centers[sl]).astype(int)
        tl = 2*tl + (scaled.targets >= centers[tl]).astype(int)

        d += 1

    return aljpy.dotdict(
        sources=sl, 
        targets=tl, 
        depth=d)

def uplift_coeffs(cheb):
    shifts = chebyshev.cartesian_product([-.5, +.5], cheb.D)
    children = shifts[..., None, :] + cheb.nodes/2
    S = cheb.similarity(cheb.nodes, children)
    return np.moveaxis(S, 0, -2)

def pushdown_coeffs(cheb):
    shifts = chebyshev.cartesian_product([-.5, +.5], cheb.D)
    children = shifts[..., None, :] + cheb.nodes/2
    S = cheb.similarity(children, cheb.nodes)
    return S

def weights(scaled, cheb, leaves):
    loc = scaled.sources * 2**leaves.depth - leaves.sources
    S = cheb.similarity(cheb.nodes, 2*loc-1)
    Ws = [leaf_sum(leaves.sources, (S*scaled.charges).T, leaves.depth)]

    coeffs = uplift_coeffs(cheb)
    dot_dims = (
        list(range(1, 2*cheb.D, 2)) + [-1],
        list(range(cheb.D)) + [-1])
    for d in reversed(range(leaves.depth)):
        exp_dims = sum([(s//2, 2) for s in Ws[-1].shape[:-1]], ())
        W_exp = Ws[-1].reshape(*exp_dims, -1)
        Ws.append(np.tensordot(W_exp, coeffs, axes=dot_dims))
    return np.array(list(reversed(Ws)))

def parent_child_format(W, D):
    width = W.shape[0]
    tail = W.shape[D:]

    Wpc = W.reshape((width//2, 2)*D + tail)
    Wpc = Wpc.transpose(
        [2*d for d in range(D)] + 
        [2*d+1 for d in range(D)] + 
        [d for d in range(2*D, Wpc.ndim)])
    return Wpc

def independent_format(Wpc, D):
    width = Wpc.shape[0]
    tail = Wpc.shape[2*D:]

    W = Wpc.transpose(
        sum([[d, D+d] for d in range(D)], []) +
        [d for d in range(2*D, Wpc.ndim)])
    W = W.reshape((2*width,)*D + tail)
    return W
    
def offset_slices(width, D):
    for offset in chebyshev.flat_cartesian_product([-1, 0, 1], D):
        first = tuple(slice(max(+o, 0), min(o+width, width)) for o in offset)
        second = tuple(slice(max(-o, 0), min(-o+width, width)) for o in offset)
        yield offset, first, second

def nephew_vectors(offset, cheb):
    D = cheb.D

    pos = chebyshev.cartesian_product([0, 1], D)[..., None, :]
    nodes = pos + (cheb.nodes + 1)/2

    child_nodes = nodes[(...,)+(None,)*(D+1)+(slice(None),)]
    nephew_nodes = (2*offset + nodes)[(None,)*(D+1)]
    node_vectors = nephew_nodes - child_nodes

    child_pos = pos[(...,)+(None,)*(D+1)+(slice(None),)]
    nephew_pos = (2*offset + pos)[(None,)*(D+1)]
    pos_vectors = nephew_pos - child_pos

    return node_vectors, pos_vectors

def interactions(W, scaled, cheb):
    """
    Input: (parent index)*D x (child offset)*D x (child node)
    Output: (parent index)*D x (child offset)*D x (child node)
    Kernel: [(neighbour offset)*D] x (child offset)*D x (child_node) x (nephew offset)*D x (nephew node)
    """
    if W.dtype == object:
        return np.array([interactions(w, scaled, cheb) for w in W])
    if W.shape[0] == 1:
        return np.zeros_like(W)

    D, N = cheb.D, cheb.N
    width = W.shape[0]

    dot_dims = (
        np.arange(D, 2*D+1),
        np.arange(D+1, 2*D+2))

    Wpc = parent_child_format(W, D)
    ixns = np.zeros_like(Wpc)
    for offset, fst, snd in offset_slices(width//2, D):
        node_vecs, pos_vecs = nephew_vectors(offset, cheb)
        K = KERNEL(np.zeros_like(node_vecs), scaled.scale*node_vecs/width)
        K = np.where((abs(pos_vecs) <= 1).all(-1), 0, K)
        ixns[snd] += np.tensordot(Wpc[fst], K, dot_dims)
    ixns = independent_format(ixns, D)

    return ixns

def far_field(ixns, cheb):
    N, D = cheb.N, cheb.D
    fs = np.empty_like(ixns)
    fs[0] = ixns[0]

    dot_dims = (D, D+1)
    coeffs = pushdown_coeffs(cheb)
    for d in range(1, len(ixns)):
        pushed = np.tensordot(fs[d-1], coeffs, dot_dims)
        dims = sum([(i, D+i) for i in range(D)], ()) + (2*D,)
        pushed = pushed.transpose(dims)

        width = 2*fs[d-1].shape[0]
        pushed = pushed.reshape((width,)*D + (N**D,))

        fs[d] = pushed + ixns[d]
    return fs

@numba.jit(nopython=True)
def _group(linear, depth, D, cutoff):
    counts = np.full(2**(depth*D), 0)
    indices = np.full((2**(depth*D), cutoff), -1)
    for i, idx in enumerate(linear):
        indices[idx, counts[idx]] = i
        counts[idx] += 1
    return indices

def group(leaves, depth, cutoff):
    D = leaves.shape[1]
    bases = (2**depth)**(np.arange(D))
    # Numba can't handle arrays of varying dimension, so here we transform
    # to linear indices and then return to subscripts afterwards
    linear = (leaves*bases).sum(-1)
    indices = _group(linear, depth, D, cutoff)
    return indices.reshape((2**depth,)*D + (cutoff,))
    
def values(fs, scaled, leaves, cheb, cutoff):
    loc = scaled.targets * 2**leaves.depth - leaves.targets
    S = cheb.similarity(2*loc-1, cheb.nodes)
    f = (S*fs[-1][tuple(leaves.targets.T)]).sum(-1)

    groups = aljpy.dotdict(
        sources=group(leaves.sources, leaves.depth, cutoff),
        targets=group(leaves.targets, leaves.depth, cutoff))

    sources, targets = scaled.scale*scaled.sources, scaled.scale*scaled.targets
    for _, fst, snd in offset_slices(2**leaves.depth, cheb.D):
        source_group = groups.sources[fst]
        target_group = groups.targets[snd]
        K = KERNEL(
            targets[target_group][..., :, None, :], 
            sources[source_group][..., None, :, :])
        charges = scaled.charges[source_group]*(source_group > -1)
        f_group = (K*charges[..., None, :]).sum(-1)

        f[target_group[target_group > -1]] += f_group[target_group > -1]
    
    return f

def solve(prob, cutoff=5):
    cheb = chebyshev.Chebyshev(10, prob.sources.shape[1])

    scaled = scale(prob)
    leaves = tree_leaves(scaled)

    Ws = weights(scaled, cheb, leaves)
    ixns = interactions(Ws, scaled, cheb)
    fs = far_field(ixns, cheb)
    v = values(fs, scaled, leaves, cheb, cutoff)

    return v

    
def run():
    prob = test.random_problem(S=100, T=100, D=2)

    v = solve(prob)

    np.testing.assert_allclose(v, test.solution(prob))