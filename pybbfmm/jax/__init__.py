import aljpy
from . import test, chebyshev
from jax import numpy as np
import jax
from functools import partial
import numpy as onp

KERNEL = jax.jit(test.quad_kernel)
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
    totals = jax.ops.index_add(totals, tuple(l.T), x)
    return totals

def leaf_centers(d):
    return (1/2 + np.arange(2**d))/2**d

def tree_leaves(scaled, cutoff=5):
    D = scaled.sources.shape[1]
    sl = np.zeros((len(scaled.sources), D), dtype=np.int32)
    tl = np.zeros((len(scaled.targets), D), dtype=np.int32)

    #TODO: You can probably get very smart about this and just convert the xs
    # to ints and look at their binary representation.
    d = 0
    while True:
        s_done = leaf_sum(sl, 1, d).max() <= cutoff
        t_done = leaf_sum(tl, 1, d).max() <= cutoff
        if s_done and t_done:
            break

        centers = leaf_centers(d)
        sl = 2*sl + (scaled.sources >= centers[sl]).astype(np.int32)
        tl = 2*tl + (scaled.targets >= centers[tl]).astype(np.int32)

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
    return list(reversed(Ws))

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
    if isinstance(W, list):
        return [interactions(w, scaled, cheb) for w in W]
    if W.shape[0] == 1:
        return np.zeros_like(W)

    D, N = cheb.D, cheb.N
    width = W.shape[0]

    dot_dims = (
        tuple(range(D, 2*D+1)),
        tuple(range(D+1, 2*D+2)))

    # Input: (parent index)*D x (child offset)*D x (child node)
    # Output: (parent index)*D x (child offset)*D x (child node)
    # Kernel: [(neighbour offset)*D] x (child offset)*D x (child_node) x (nephew offset)*D x (nephew node)
    Wpc = parent_child_format(W, D)
    ixns = np.zeros_like(Wpc)
    for offset, fst, snd in offset_slices(width//2, D):
        node_vecs, pos_vecs = nephew_vectors(offset, cheb)
        K = KERNEL(np.zeros_like(node_vecs), scaled.scale*node_vecs/width)
        K = np.where((abs(pos_vecs) <= 1).all(-1), 0, K)
        ixns = jax.ops.index_add(ixns, snd, np.tensordot(Wpc[fst], K, dot_dims))
    ixns = independent_format(ixns, D)

    return ixns

def far_field(ixns, cheb):
    N, D = cheb.N, cheb.D
    fs = [None for _ in ixns]
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

def linear_index(subscripts, depth):
    D = subscripts.shape[-1]
    bases = (2**depth)**np.arange(D)
    linear = (subscripts*bases).sum(-1)
    return linear

def counts_of(linear):
    argsort = np.argsort(linear)
    ordered = linear[argsort]
    last  = np.concatenate([ordered[1:] != ordered[:-1], np.array([True])])
    cum_replicas = np.concatenate([
        np.asarray([-1,]),
        np.arange(len(linear))[last]])
    n_replicas = np.diff(cum_replicas)

    steps = np.ones(len(linear)-1, dtype=n_replicas.dtype)
    steps = jax.ops.index_add(steps, last[:-1], -n_replicas[:-1])
    steps = np.concatenate([np.array([0]), steps])
    ordered_index = np.cumsum(steps)

    counts = np.empty_like(ordered_index)
    counts = jax.ops.index_update(counts, argsort, ordered_index)

    return counts

def group(leaves, depth, cutoff):
    linear = linear_index(leaves, depth)
    counts = counts_of(linear)

    D = leaves.shape[-1]
    indices = np.full((2**(depth*D), cutoff), -1)
    indices = jax.ops.index_update(indices, (linear, counts), np.arange(len(linear)))

    return indices.reshape((2**depth,)*D + (cutoff,))

def neighbours(groups):
    cutoff = groups.sources.shape[-1]
    pairs = np.stack([
        np.repeat(groups.sources[..., None, :], cutoff, -2),
        np.repeat(groups.targets[..., :, None], cutoff, -1)], -1)
    return pairs[(pairs > -1).all(-1)]

def near_field(scaled, leaves, cutoff):
    sources, targets = scaled.scale*scaled.sources, scaled.scale*scaled.targets

    groups = aljpy.dotdict(
        sources=group(leaves.sources, leaves.depth, cutoff),
        targets=group(leaves.targets, leaves.depth, cutoff))
    source_idxs, target_idxs = neighbours(groups).T 

    K = KERNEL(sources[source_idxs], targets[target_idxs])

    totals = np.zeros(len(targets))
    totals = jax.ops.index_add(totals, target_idxs, K*scaled.charges[source_idxs])

    return totals

def values(fs, scaled, leaves, cheb, cutoff):
    n = near_field(scaled, leaves, cutoff)

    loc = scaled.targets * 2**leaves.depth - leaves.targets
    S = cheb.similarity(2*loc-1, cheb.nodes)
    f = (S*fs[-1][tuple(leaves.targets.T)]).sum(-1)
    
    return f + n

def solve(prob, N=4, cutoff=8):
    cheb = chebyshev.Chebyshev(N, prob.sources.shape[1])

    scaled = scale(prob)
    leaves = tree_leaves(scaled, cutoff=cutoff)

    ws = weights(scaled, cheb, leaves)
    ixns = interactions(ws, scaled, cheb)
    fs = far_field(ixns, cheb)
    v = values(fs, scaled, leaves, cheb, cutoff)

    return v

def run():
    N = 4
    cutoff = 5

    prob = test.random_problem(S=100, T=100, D=2)
    cheb = chebyshev.Chebyshev(N, prob.sources.shape[1])

    scaled = scale(prob)
    leaves = tree_leaves(scaled, cutoff=cutoff)
    ws = weights(scaled, cheb, leaves)
    ixns = interactions(ws, scaled, cheb)