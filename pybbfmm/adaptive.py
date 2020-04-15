import aljpy
from aljpy import arrdict
from . import test, chebyshev, orthantree
import torch

KERNEL = test.quad_kernel
EPS = 1e-2

def limits(prob):
    points = torch.cat([prob.sources, prob.targets])
    return torch.stack([points.min(0).values - EPS, points.max(0).values + EPS])

def scale(prob):
    lims = limits(prob)
    mid = (lims[0] + lims[1])/2
    scale = (lims[1] - lims[0])/2
    return aljpy.dotdict(
        limits=lims,
        scale=scale,
        sources=(prob.sources - mid)/scale,
        charges=prob.charges,
        targets=(prob.targets - mid)/scale)

def accumulate(indices, vals, length):
    totals = vals.new_zeros((length,) + vals.shape[1:])
    totals.index_add_(0, indices, vals)
    return totals

def inner_product(A, B):
    """Given an array of pairs (a, p) and another of pairs (q, b), returns all pairs (a, c)
    such that for there's some pair (a, r) in the first array and another (r, b) in the second.
    
    Which is to say, it's an inner join. As a bonus, it automatically adds outer columns of 
    indices to anything 1D passed.
    """
    if A.ndim == 1:
        A = torch.stack([torch.arange(len(A), dtype=A.dtype, device=A.device), A], -1)
    if B.ndim == 1:
        B = torch.stack([B, torch.arange(len(B), dtype=B.dtype, device=B.device)], -1)

    A_order = torch.argsort(A[:, 1])
    A_sorted = A[A_order]
    A_unique, A_inv, A_counts = torch.unique(A_sorted[:, 1], return_inverse=True, return_counts=True)

    B_order = torch.argsort(B[:, 0])
    B_sorted = B[B_order]
    B_unique, B_inv, B_counts = torch.unique(B_sorted[:, 0], return_inverse=True, return_counts=True)

    C_unique, C_inv = torch.unique(torch.cat([A_unique, B_unique]), return_inverse=True)
    A_unique_inv, B_unique_inv = C_inv[:len(A_unique)], C_inv[len(A_unique):]

    CA_counts = torch.zeros_like(C_unique)
    CA_counts[A_unique_inv] = A_counts

    CB_counts = torch.zeros_like(C_unique)
    CB_counts[B_unique_inv] = B_counts

    pairs = []
    for A_reps in range(1, CA_counts.max()+1):
        for B_reps in range(1, CB_counts.max()+1):
            mask = (CA_counts == A_reps) & (CB_counts == B_reps)
            
            A_vals = A_sorted[mask[A_unique_inv[A_inv]], 0]
            B_vals = B_sorted[mask[B_unique_inv[B_inv]], 1]

            pairs.append(torch.stack([
                torch.repeat_interleave(A_vals, B_reps),
                torch.repeat_interleave(B_vals, A_reps)], -1))
    pairs = torch.cat(pairs)

    return pairs

def uplift_coeffs(cheb):
    shifts = torch.tensor([-.5, +.5], device=cheb.device)
    shifts = chebyshev.cartesian_product(shifts, cheb.D)
    children = shifts[..., None, :] + cheb.nodes/2
    return cheb.similarity(cheb.nodes, children)

def weights(scaled, cheb, tree, indices):
    loc = 2**tree.depths[indices.sources, None]*(scaled.sources - tree.centers[indices.sources])
    S = cheb.similarity(loc, cheb.nodes)
    W = accumulate(indices.sources, S, len(tree.id))

    coeffs = uplift_coeffs(cheb)
    dot_dims = (list(range(1, cheb.D+2)), list(range(1, cheb.D+2)))

    parents = tree.parents[indices.sources]
    while (parents >= 0).any():
        parents = parents[parents >= 0]
        W[parents] = torch.tensordot(W[tree.children[parents]], coeffs, dot_dims)
        parents = tree.parents[parents] 

    return W

def node_points(scaled, cheb, tree, indices):
    return scaled.scale*(cheb.nodes[None]/2**tree.depths[indices, None, None] + tree.centers[indices, None, :])

def interactions(W, scaled, cheb, tree, indices, lists):
    # depthscale = 2**(tree.depths.max()-1)
    # vectors = (depthscale*(tree.centers[lists.v[:, 0]] - tree.centers[lists.v[:, 1]])).int()

    # D = vectors.shape[1]
    # base = (2*depthscale)**torch.arange(D, device=vectors.device, dtype=torch.int32)
    # vector_id = ((2*depthscale + vectors)*base).sum(-1)
    # unique_id, inv = torch.unique(vector_id, return_inverse=True)
    ixns = torch.zeros_like(W)

    # V contributions
    nodes = node_points(scaled, cheb, tree, lists.v)
    K = KERNEL(nodes[:, 0, :, None], nodes[:, 1, None, :])
    Wv = torch.einsum('ijk,ik->ij', K, W[lists.v[:, 1]])
    ixns.index_add_(0, lists.v[:, 0], Wv)

    # X contributions
    pairs = inner_product(lists.x, indices.sources)
    K = KERNEL(node_points(scaled, cheb, tree, pairs[:, 0]), scaled.sources[pairs[:, 1], None, :])
    Wx = K*scaled.charges[pairs[:, 1], None]
    ixns.index_add(0, pairs[:, 0], Wx)

def run():
    torch.random.manual_seed(1)
    prob = aljpy.dotdict(
        sources=torch.tensor([[-.4, .4], [-.8, .8]]),
        charges=torch.tensor([1., 1.]),
        targets=torch.empty((0, 2)))

    prob = test.random_problem(S=100, T=100)

    scaled = scale(prob)
    cheb = chebyshev.Chebyshev(4, scaled.sources.shape[1], device='cuda')

    tree, indices = orthantree.build(scaled)
    lists = orthantree.pairlists(tree)

    W = weights(scaled, cheb, tree, indices)