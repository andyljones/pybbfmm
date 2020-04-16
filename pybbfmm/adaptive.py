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

def linear_coord(coords, max_depth):
    D = coords.shape[1]
    dims = torch.arange(D-1, -1, -1, dtype=coords.dtype, device=coords.device)
    bases = 2**(max_depth*dims)
    return (coords*bases).sum(-1), bases

def integer_coords(vectors, max_depth):
    return (2**(max_depth-2)*(vectors + 2)).int()

def float_coords(vectors, max_depth):
    return vectors.float()/2**(max_depth-2) - 2

def conventional_coords(linear, bases):
    coords = []
    for b in bases[:-1]:
        coords.append(linear // b)
        linear = linear - b*coords[-1]
    return torch.stack(coords + [linear], -1)

def unique_vectors(vectors, max_depth):
    coords = integer_coords(vectors, max_depth)
    linear, bases = linear_coord(coords, max_depth)

    unique, inv = torch.unique(linear, return_inverse=True)
    unique = conventional_coords(unique, bases)
    return float_coords(unique, max_depth), inv

def node_locations(scaled, cheb, tree, indices):
    return scaled.scale*(cheb.nodes[None]/2**tree.depths[indices, None, None] + tree.centers[indices, None, :])

def v_interactions(W, scaled, cheb, tree, lists):
    # vectors = tree.centers[lists.v[:, 0]] - tree.centers[lists.v[:, 1]]
    # vectors, inv = unique_vectors(vectors, tree.depths.max())

    nodes = node_locations(scaled, cheb, tree, lists.v)
    K = KERNEL(nodes[:, 0, :, None], nodes[:, 1, None, :])
    ixns = torch.einsum('ijk,ik->ij', K, W[lists.v[:, 1]]) if len(lists.v) > 0 else W[:0]
    return sets.accumulate(lists.v[:, 0], ixns, len(tree.id))

def x_interactions(scaled, cheb, tree, indices, lists):
    pairs = sets.inner_join(lists.x, sets.right_index(indices.sources))
    K = KERNEL(node_locations(scaled, cheb, tree, pairs[:, 0]), scaled.scale*scaled.sources[pairs[:, 1], None, :])
    ixns = K*scaled.charges[pairs[:, 1], None]
    return sets.accumulate(pairs[:, 0], ixns, len(tree.id))

def w_interactions(W, scaled, cheb, tree, indices, lists):
    pairs = sets.inner_join(sets.left_index(indices.targets), lists.w)
    K = KERNEL(scaled.scale*scaled.targets[pairs[:, 0], None, :], node_locations(scaled, cheb, tree, pairs[:, 1]))
    ixns = torch.einsum('ij,ij->i', K, W[pairs[:, 1]]) if len(pairs) > 0 else scaled.charges[:0]
    return sets.accumulate(pairs[:, 0], ixns, len(indices.targets))

def u_interactions(scaled, indices, lists):
    pairs = sets.inner_join(lists.u, sets.right_index(indices.sources))
    pairs = sets.inner_join(sets.left_index(indices.targets), pairs)
    K = KERNEL(scaled.scale*scaled.targets[pairs[:, 0]], scaled.scale*scaled.sources[pairs[:, 1]])
    return sets.accumulate(pairs[:, 0], K*scaled.charges[pairs[:, 1]], len(scaled.targets))

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

def target_far_field(F, scaled, cheb, tree, indices):
    loc = 2**tree.depths[indices.targets, None]*(scaled.targets - tree.centers[indices.targets])
    S = cheb.similarity(loc, cheb.nodes)
    return torch.einsum('ij,ij->i', S, F[indices.targets]) if len(indices.targets) > 0 else scaled.charges[:0]

def solve(prob):
    cheb = chebyshev.Chebyshev(4, prob.sources.shape[1], device='cuda')
    scaled = scale(prob)
    tree, indices = orthantree.orthantree(scaled)
    lists = orthantree.interaction_lists(tree)

    W = weights(scaled, cheb, tree, indices)

    v = v_interactions(W, scaled, cheb, tree, lists)
    x = x_interactions(scaled, cheb, tree, indices, lists)

    F = far_field(W, v, x, cheb, tree)
    f = target_far_field(F, scaled, cheb, tree, indices)

    w = w_interactions(W, scaled, cheb, tree, indices, lists)
    u = u_interactions(scaled, indices, lists)

    return f + w + u

def run():
    prob = test.random_problem(S=100, T=100)
    soln = solve(prob)
    ref = test.solve(prob)
    (soln - ref).pow(2).sum()/ref.pow(2).sum()
