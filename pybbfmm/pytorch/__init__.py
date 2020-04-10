import aljpy
from aljpy import arrdict
from . import test, chebyshev
import numpy as np
import torch

KERNEL = test.quad_kernel
EPS = 1e-2

def limits(prob):
    points = torch.cat([prob.sources, prob.targets])
    return torch.stack([points.min(0).values - EPS, points.max(0).values + EPS])

@test.report_memory
def scale(prob):
    lims = limits(prob)
    lower, scale = lims[0], lims[1] - lims[0]
    return aljpy.dotdict(
        limits=lims,
        scale=scale,
        sources=(prob.sources - lower)/scale,
        charges=prob.charges,
        targets=(prob.targets - lower)/scale)

def accumulate(subscripts, vals, shape):
    flipped = torch.flip(torch.tensor(shape, device=subscripts.device), (0,))
    bases = torch.flip(torch.cumprod(flipped, 0)/flipped, (0,))
    indices = (subscripts*bases).sum(-1)
    
    totals = vals.new_zeros((np.prod(shape),) + vals.shape[1:])
    totals.index_add_(0, indices, vals)

    return totals.reshape(shape + vals.shape[1:])

def value_counts(subscripts, shape):
    vals = subscripts.new_ones((len(subscripts),), dtype=torch.int32)
    return accumulate(subscripts, vals, shape)

def leaf_centers(d):
    return (1/2 + torch.arange(2**d))/2**d

@test.report_memory
def tree_leaves(scaled, cutoff=5):
    D = scaled.sources.shape[1]
    sl = torch.zeros_like(scaled.sources, dtype=torch.long)
    tl = torch.zeros_like(scaled.targets, dtype=torch.long)

    #TODO: You can probably get very smart about this and just convert the xs
    # to ints and look at their binary representation.
    d = 0
    while True:
        s_done = value_counts(sl, (2**d,)*D).max() <= cutoff
        t_done = value_counts(tl, (2**d,)*D).max() <= cutoff
        if s_done and t_done:
            break

        centers = leaf_centers(d).to(sl.device)
        sl = 2*sl + (scaled.sources >= centers[sl]).long()
        tl = 2*tl + (scaled.targets >= centers[tl]).long()

        d += 1

    return aljpy.dotdict(
        sources=sl, 
        targets=tl, 
        depth=d,
        cutoff=cutoff)

def linear_index(subscripts, depth):
    D = subscripts.shape[-1]
    bases = (2**depth)**torch.arange(D, device=subscripts.device)
    linear = (subscripts*bases).sum(-1)
    return linear

def cohabiting_pairs(sources, targets, depth, cutoff):
    D = sources.shape[-1]
    max_idx = 2**(D*depth)

    source_idxs = linear_index(sources, depth)
    source_order = torch.argsort(source_idxs)
    source_sorted = source_idxs[source_order]
    source_counts = value_counts(source_sorted[:, None], (max_idx,))

    target_idxs = linear_index(targets, depth)
    target_order = torch.argsort(target_idxs)
    target_sorted = target_idxs[target_order]
    target_counts = value_counts(target_sorted[:, None], (max_idx,))

    pairs = []
    for source_count in range(1, cutoff+1):
        for target_count in range(1, cutoff+1):
            mask = (source_counts == source_count) & (target_counts == target_count)
            s = mask[source_sorted].nonzero()
            t = mask[target_sorted].nonzero()

            ps = torch.stack([
                    torch.repeat_interleave(s.reshape(mask.sum(), source_count, 1), target_count, 2),
                    torch.repeat_interleave(t.reshape(mask.sum(), 1, target_count), source_count, 1)], -1).reshape(-1, 2)
            pairs.append(ps)
    pairs = torch.cat(pairs)
    pairs = torch.stack([source_order[pairs[..., 0]], target_order[pairs[..., 1]]], -1)
    return pairs

def neighbouring_pairs(leaves):
    D = leaves.sources.shape[-1]
    pairs = []
    offsets = torch.tensor([-1, 0, +1], device=leaves.sources.device)
    for offset in chebyshev.flat_cartesian_product(offsets, D):
        offset_sources = leaves.sources + offset
        mask = ((0 <= offset_sources) & (offset_sources < 2**leaves.depth)).all(-1)
        masked_pairs = cohabiting_pairs(offset_sources[mask], leaves.targets, leaves.depth, leaves.cutoff)
        pairs.append(torch.stack([
            mask.nonzero()[:, 0][masked_pairs[..., 0]],
            masked_pairs[..., 1]], -1))
    return torch.cat(pairs)

def uplift_coeffs(cheb):
    shifts = torch.tensor([-.5, +.5], device=cheb.device)
    shifts = chebyshev.cartesian_product(shifts, cheb.D)
    children = shifts[..., None, :] + cheb.nodes/2
    S = cheb.similarity(cheb.nodes, children)
    dims = tuple(range(1, S.ndim-1)) + (0, -1)
    return S.permute(dims)

def pushdown_coeffs(cheb):
    shifts = torch.tensor([-.5, +.5], device=cheb.device)
    shifts = chebyshev.cartesian_product(shifts, cheb.D)
    children = shifts[..., None, :] + cheb.nodes/2
    S = cheb.similarity(children, cheb.nodes)
    return S

@test.report_memory
def weights(scaled, cheb, leaves):
    loc = scaled.sources * 2**leaves.depth - leaves.sources
    S = cheb.similarity(2*loc-1, cheb.nodes)
    Ws = [accumulate(leaves.sources, scaled.charges[:, None]*S, (2**leaves.depth,)*cheb.D)]

    coeffs = uplift_coeffs(cheb)
    dot_dims = (
        list(range(1, 2*cheb.D, 2)) + [-1],
        list(range(cheb.D)) + [-1])
    for d in reversed(range(leaves.depth)):
        exp_dims = sum([(s//2, 2) for s in Ws[-1].shape[:-1]], ())
        W_exp = Ws[-1].reshape(*exp_dims, -1)
        Ws.append(torch.tensordot(W_exp, coeffs, dot_dims))
    return list(reversed(Ws))

def parent_child_format(W, D):
    width = W.shape[0]
    tail = W.shape[D:]

    Wpc = W.reshape((width//2, 2)*D + tail)
    Wpc = Wpc.permute(
        [2*d for d in range(D)] + 
        [2*d+1 for d in range(D)] + 
        [d for d in range(2*D, Wpc.ndim)])
    return Wpc

def independent_format(Wpc, D):
    width = Wpc.shape[0]
    tail = Wpc.shape[2*D:]

    W = Wpc.permute(
        sum([[d, D+d] for d in range(D)], []) +
        [d for d in range(2*D, Wpc.ndim)])
    W = W.reshape((2*width,)*D + tail)
    return W
    
def offset_slices(width, D, device=None):
    offset = torch.tensor([-1, 0, 1], device=device)
    for offset in chebyshev.flat_cartesian_product(offset, D):
        first = tuple(slice(max( o, 0), min(o+width, width)) for o in offset)
        second = tuple(slice(max(-o, 0), min(-o+width, width)) for o in offset)
        yield offset, first, second

def nephew_vectors(offset, cheb):
    D = cheb.D

    pos = torch.tensor([0, 1], device=cheb.device)
    pos = chebyshev.cartesian_product(pos, D)[..., None, :]
    nodes = pos + (cheb.nodes + 1)/2

    child_nodes = nodes[(...,)+(None,)*(D+1)+(slice(None),)]
    nephew_nodes = (2*offset + nodes)[(None,)*(D+1)]
    node_vectors = nephew_nodes - child_nodes

    child_pos = pos[(...,)+(None,)*(D+1)+(slice(None),)]
    nephew_pos = (2*offset + pos)[(None,)*(D+1)]
    pos_vectors = nephew_pos - child_pos

    return node_vectors, pos_vectors

@test.report_memory
def interactions(W, scaled, cheb):
    if isinstance(W, list):
        return [interactions(w, scaled, cheb) for w in W]
    if W.shape[0] == 1:
        return torch.zeros_like(W)

    D, N = cheb.D, cheb.N
    width = W.shape[0]

    dot_dims = (
        tuple(range(D, 2*D+1)),
        tuple(range(D+1, 2*D+2)))

    # Input: (parent index)*D x (child offset)*D x (child node)
    # Output: (parent index)*D x (child offset)*D x (child node)
    # Kernel: [(neighbour offset)*D] x (child offset)*D x (child_node) x (nephew offset)*D x (nephew node)
    Wpc = parent_child_format(W, D)
    ixns = torch.zeros_like(Wpc)
    for offset, fst, snd in offset_slices(width//2, D, cheb.device):
        node_vecs, pos_vecs = nephew_vectors(offset, cheb)
        K = KERNEL(torch.zeros_like(node_vecs), scaled.scale*node_vecs/width)
        K = torch.where((abs(pos_vecs) <= 1).all(-1), torch.zeros_like(K), K)
        ixns[snd] += torch.tensordot(Wpc[fst], K, dot_dims)
    ixns = independent_format(ixns, D)

    return ixns

@test.report_memory
def far_field(ixns, cheb):
    N, D = cheb.N, cheb.D
    fs = [None for _ in ixns]
    fs[0] = ixns[0]

    dot_dims = ((D,), (D+1,))
    coeffs = pushdown_coeffs(cheb)
    for d in range(1, len(ixns)):
        pushed = torch.tensordot(fs[d-1], coeffs, dot_dims)
        dims = sum([(i, D+i) for i in range(D)], ()) + (2*D,)
        pushed = pushed.permute(dims)

        width = 2*fs[d-1].shape[0]
        pushed = pushed.reshape((width,)*D + (N**D,))

        fs[d] = pushed + ixns[d]
    return fs

@test.report_memory
def target_far_field(fs, scaled, leaves, cheb):
    loc = scaled.targets * 2**leaves.depth - leaves.targets
    S = cheb.similarity(2*loc-1, cheb.nodes)
    f = (S*fs[-1][tuple(leaves.targets.T)]).sum(-1)
    return f


@test.report_memory
def near_field(scaled, pairs):
    source_idxs, target_idxs = pairs.T
    sources = (scaled.scale*scaled.sources)[source_idxs]
    targets = (scaled.scale*scaled.targets)[target_idxs]
    charges = scaled.charges[source_idxs]

    totals = scaled.charges.new_zeros(len(scaled.targets))
    totals.index_add_(0, target_idxs, KERNEL(sources, targets)*charges)

    return totals

@test.report_memory
def prepare(prob, N=4, cutoff=8):
    r = aljpy.dotdict()
    D = prob.sources.shape[1]
    r['cheb'] = chebyshev.Chebyshev(N, D, prob.sources.device)
    r['scaled'] = scale(prob)
    r['leaves'] = tree_leaves(r.scaled, cutoff)
    r['pairs'] = neighbouring_pairs(r.leaves)
    return r

@test.report_memory
def solve(prob, prep=None, **kwargs):
    prep = prepare(prob) if prep is None else prep

    ws = weights(prep.scaled, prep.cheb, prep.leaves)
    ixns = interactions(ws, prep.scaled, prep.cheb)
    fs = far_field(ixns, prep.cheb)
    f = target_far_field(fs, prep.scaled, prep.leaves, prep.cheb)
    n = near_field(prep.scaled, prep.pairs)
    v = f + n

    return v.cpu().numpy()

def run():
    prob = test.random_problem(S=100, T=100, D=2)
    soln = solve(prob)

def benchmark(maxsize=5e6, repeats=5):
    import pandas as pd

    result = {}
    for N in np.logspace(1, np.log10(maxsize), 10, dtype=int):
        print(f'Timing {N}')
        for r in range(repeats):
            prob = test.random_problem(S=N, T=N, D=2)
            solve(prob)
            with aljpy.timer() as bbfmm:
                solve(prob)
            result[N, r] = bbfmm.time()
    result = pd.Series(result)

    import matplotlib.pyplot as plt
    with plt.style.context('seaborn-poster'):
        ax = result.groupby(level=0).mean().plot()
        ax.set_title('N=5, D=2')
        ax.set_xlabel('n points')
        ax.set_ylabel('average runtime')

    return result