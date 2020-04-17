import aljpy
from aljpy import arrdict
from . import test, chebyshev, orthantree, sets
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

def weights(scaled, cheb, tree, indices):
    loc = 2**tree.depths[indices.sources, None]*(scaled.sources - tree.centers[indices.sources])
    S = cheb.similarity(loc, cheb.nodes)
    W = sets.accumulate(indices.sources, S*scaled.charges[:, None], len(tree.id))

    coeffs = cheb.upwards_coeffs()
    dot_dims = (list(range(1, cheb.D+2)), list(range(1, cheb.D+2)))

    parents = tree.parents[indices.sources]
    while (parents >= 0).any():
        parents = parents[parents >= 0]
        W[parents] = torch.tensordot(W[tree.children[parents]], coeffs, dot_dims)
        parents = tree.parents[parents] 

    return W

def node_locations(scaled, cheb, tree, indices):
    return scaled.scale*(cheb.nodes[None]/2**tree.depths[indices, None, None] + tree.centers[indices, None, :])

def v_interactions(W, scaled, cheb, tree, scheme):
    offsets = scheme.v_vectors.offsets[:, None, None]
    scales = scaled.scale/2**scheme.v_vectors.depths[:, None, None, None]
    boxes = scales*cheb.nodes[None, None, :]
    partners = scales*(offsets + cheb.nodes[None, :, None])
    K = KERNEL(boxes, partners) 

    ixns = torch.zeros_like(W) 
    for i, k in enumerate(K):
        mask = (scheme.v_vectors.inverse == i)
        vs = scheme.lists.v[mask]
        ixns.index_add_(0, vs[:, 0], W[vs[:, 1]] @ k)
    return ixns

def x_interactions(scaled, cheb, tree, indices, scheme, chunksize=int(1e6)):
    ixns = scaled.charges.new_zeros((len(tree.id), cheb.N**cheb.D))
    chunks = (scheme.lists.x[i:i+chunksize] for i in range(0, len(scheme.lists.x), chunksize))
    for chunk in chunks:
        pairs = sets.inner_join(chunk, sets.right_index(indices.sources))
        K = KERNEL(node_locations(scaled, cheb, tree, pairs[:, 0]), scaled.scale*scaled.sources[pairs[:, 1], None, :])
        ixns.index_add_(0, pairs[:, 0], K*scaled.charges[pairs[:, 1], None])
    return ixns

def w_interactions(W, scaled, cheb, tree, indices, scheme, chunksize=int(1e6)):
    ixns = scaled.charges.new_zeros(len(scaled.targets))
    chunks = (scheme.lists.w[i:i+chunksize] for i in range(0, len(scheme.lists.w), chunksize))
    for chunk in chunks:
        pairs = sets.inner_join(sets.left_index(indices.targets), chunk)
        K = KERNEL(scaled.scale*scaled.targets[pairs[:, 0], None, :], node_locations(scaled, cheb, tree, pairs[:, 1]))
        ixns.index_add_(0, pairs[:, 0], (K*W[pairs[:, 1]]).sum(-1))
    return ixns

def u_interactions(scaled, indices, scheme, chunksize=int(1e6)):
    ixns = scaled.charges.new_zeros(len(scaled.targets))
    chunks = (scheme.lists.u[i:i+chunksize] for i in range(0, len(scheme.lists.u), chunksize))
    for chunk in chunks:
        pairs = sets.inner_join(chunk, sets.right_index(indices.sources))
        pairs = sets.inner_join(sets.left_index(indices.targets), pairs)
        K = KERNEL(scaled.scale*scaled.targets[pairs[:, 0]], scaled.scale*scaled.sources[pairs[:, 1]])
        ixns.index_add_(0, pairs[:, 0], K*scaled.charges[pairs[:, 1]])
    return ixns

def far_field(W, v, x, cheb, tree):
    F = torch.zeros_like(W)
    coeffs = cheb.downwards_coeffs()
    dot_dims = ((1,), (-1,))

    parents = tree.parents.new_tensor([0])
    while parents.nelement():
        parents = parents[~tree.terminal[parents]]
        children = tree.children[parents]
        F[children] = torch.tensordot(F[parents], coeffs, dot_dims) + v[children] + x[children]
        parents = children.flatten()
    
    return F

def target_far_field(F, scaled, cheb, tree, indices, chunksize=int(1e6)):
    potentials = scaled.charges.new_zeros(len(scaled.targets))
    for i in range(0, len(indices.targets), chunksize):
        idx_chunk = indices.targets[i:i+chunksize]
        tgt_chunk = scaled.targets[i:i+chunksize]
        loc = 2**tree.depths[idx_chunk, None]*(tgt_chunk - tree.centers[idx_chunk])
        S = cheb.similarity(loc, cheb.nodes)
        potentials[i:i+chunksize] = (S*F[idx_chunk]).sum(-1)
    return potentials

def solve(prob):
    cheb = chebyshev.Chebyshev(4, prob.sources.shape[1], device='cuda')
    scaled = scale(prob)
    tree, indices = orthantree.orthantree(scaled)
    scheme = orthantree.interaction_scheme(tree)

    W = weights(scaled, cheb, tree, indices)

    v = v_interactions(W, scaled, cheb, tree, scheme)
    x = x_interactions(scaled, cheb, tree, indices, scheme)

    F = far_field(W, v, x, cheb, tree)
    f = target_far_field(F, scaled, cheb, tree, indices)

    w = w_interactions(W, scaled, cheb, tree, indices, scheme)
    u = u_interactions(scaled, indices, scheme)

    return f + w + u

def run():
    prob = test.random_problem(S=100, T=100)
    soln = solve(prob)
    ref = test.solve(prob)
    (soln - ref).pow(2).sum()/ref.pow(2).sum()
