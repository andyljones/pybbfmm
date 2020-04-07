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
    return aljpy.dotdict(
        limits=lims,
        sources=(prob.sources - lims[0])/(lims[1] - lims[0]),
        charges=prob.charges,
        targets=(prob.targets - lims[0])/(lims[1] - lims[0]))

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

def nephew_vectors(cheb):
    # (nephew offset) + (nephew node) + (child offset) + (child node) + (D,) 
    # (3, 2)*D + (N**D,) + (2,)*D + (N**D,) + (D,)
    D, N = cheb.D, cheb.N

    child_dims = (1, 1)*D + (1,) + (2,)*D + (1,) + (D,)
    child_offsets = chebyshev.cartesian_product([0, 1], D).reshape(child_dims)

    child_node_dims = (1, 1)*D + (1,) + (1,)*D + (N**D,) + (D,)
    child_nodes = (cheb.nodes/2 + 1/2).reshape(child_node_dims)

    nephew_dims = (3, 2)*D + (1,) + (1,)*D + (1,) + (D,)
    nephew_offsets = chebyshev.cartesian_product(np.arange(-2, 4), D).reshape(nephew_dims)

    nephew_node_dims = (1, 1)*D + (N**D,) + (1,)*D + (1,) + (D,)
    nephew_nodes = (cheb.nodes/2 + 1/2).reshape(nephew_node_dims)

    vectors = ((nephew_offsets + nephew_nodes) - (child_offsets + child_nodes))
    is_neighbour = (abs(nephew_offsets - child_offsets) <= 1).all(-1)

    return vectors, is_neighbour

def interactions(W, scaled, cheb):
    if W.dtype == object:
        return np.array([interactions(w, scaled, cheb) for w in W])
    if W.shape[0] == 1:
        return np.zeros_like(W)

    D, N = cheb.D, cheb.N
    width = W.shape[0]

    vectors, is_neighbour = nephew_vectors(cheb)
    vectors = (scaled.limits[1] - scaled.limits[0])/width*vectors

    nephew_kernel = KERNEL(np.zeros_like(vectors), vectors)
    interaction_kernel = np.where(is_neighbour, 0, nephew_kernel)

    mirrored = interaction_kernel[(slice(None, None, -1), slice(None))*D]

    W_dims = (width//2+2, 2)*D + (N**D,) + (1,)*D + (1,)
    Wp = np.pad(W, (((2, 2),)*D + ((0, 0),)))
    Wp = Wp.reshape(W_dims)

    ixns = scipy.signal.fftconvolve(
        Wp,
        mirrored,
        mode='valid',
        axes=np.arange(0, 2*D, 2)
    ).sum(tuple(2*i+1 for i in range(D)) + (2*D,), keepdims=True)

    squeeze = tuple(2*d+1 for d in range(D)) + (2*D,)
    ixns = ixns.squeeze(squeeze)

    axes = sum([(i, D+i) for i in range(D)], ()) + (2*D,)
    ixns = ixns.transpose(axes)

    ixns = ixns.reshape((width,)*D + (N**D,))

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

def offset_slices(depth, D):
    width = 2**depth
    for offset in chebyshev.flat_cartesian_product([-1, 0, 1], D):
        first = tuple(slice(max(+o, 0), min(o+width, width)) for o in offset)
        second = tuple(slice(max(-o, 0), min(-o+width, width)) for o in offset)
        yield first, second
    

def values(fs, scaled, leaves, cheb, cutoff):
    loc = scaled.targets * 2**leaves.depth - leaves.targets
    S = cheb.similarity(2*loc-1, cheb.nodes)
    f = (S*fs[-1][tuple(leaves.targets.T)]).sum(-1)

    groups = aljpy.dotdict(
        sources=group(leaves.sources, leaves.depth, cutoff),
        targets=group(leaves.targets, leaves.depth, cutoff))

    scale = scaled.limits[1] - scaled.limits[0]
    sources, targets = scale*scaled.sources, scale*scaled.targets
    for fst, snd in offset_slices(leaves.depth, cheb.D):
        source_group = groups.sources[fst]
        target_group = groups.targets[snd]
        K = KERNEL(
            targets[target_group][..., :, None, :], 
            sources[source_group][..., None, :, :])
        charges = scaled.charges[source_group]*(source_group > -1)
        f_group = (K*charges[..., None, :]).sum(-1)

        f[target_group[target_group > -1]] += f_group[target_group > -1]
    
    return f

    
def run():
    prob = test.random_problem(S=100, T=100, D=2)

    cheb = chebyshev.Chebyshev(10, prob.sources.shape[1])

    cutoff = 5
    scaled = scale(prob)
    leaves = tree_leaves(scaled)

    Ws = weights(scaled, cheb, leaves)
    ixns = interactions(Ws, scaled, cheb)
    fs = far_field(ixns, cheb)
    v = values(fs, scaled, leaves, cheb, cutoff)

    np.testing.assert_allclose(v, test.solution(prob))

    # Validation
    root = tree.build_tree(prob, cheb=cheb)
    root.set_weights()
    tree.set_interactions(root)
    root.set_far_field()
    v_ref = root.values()

    np.testing.assert_allclose(root.W, Ws[0][0, 0])

    for i, j, k, l in chebyshev.flat_cartesian_product([0, 1], 4):
        np.testing.assert_allclose(
            root.children[i, j].children[k, l].f,
            ixns[2][2*i+k, 2*j+l])

        for m, n in chebyshev.flat_cartesian_product([0, 1], 2):
            np.testing.assert_allclose(
                root.children[i, j].children[k, l].children[m, n].f,
                fs[3][4*i+2*k+m, 4*j+2*l+n])
    
    np.testing.assert_allclose(v_ref, v)

