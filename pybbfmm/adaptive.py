import aljpy
from aljpy import arrdict
from . import test, chebyshev
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def tree_indices(scaled, cutoff=5):
    #TODO: Well this is a travesty of incomprehensibility. Verify it then explain yourself.
    D = scaled.sources.shape[1]

    points = torch.cat([scaled.sources, scaled.targets])
    indices = points.new_zeros((len(points),), dtype=torch.long)

    tree = arrdict.arrdict(
        parents=indices.new_full((1,), -1),
        depths=indices.new_zeros((1,)),
        centers=points.new_zeros((1, D)),
        terminal=indices.new_ones((1,), dtype=torch.bool),
        children=indices.new_full((1,) + (2,)*D, -1),
        childtypes=indices.new_zeros((1, D)))

    bases = 2**torch.flip(torch.arange(D, device=indices.device), (0,))
    subscript_offsets = chebyshev.cartesian_product(torch.tensor([0, 1], device=indices.device), D)
    center_offsets = chebyshev.cartesian_product(torch.tensor([-1, +1], device=indices.device), D)

    depth = 0
    while True:
        used, used_inv, counts = torch.unique(indices, return_inverse=True, return_counts=True)
        tree.terminal[used] = (counts <= cutoff)
        
        used_is_active = ~tree.terminal[used]
        point_is_active = used_is_active[used_inv]
        if not point_is_active.any():
            break

        depth += 1
        
        active = used[used_is_active]
        active_inv = (used_is_active.cumsum(0) - used_is_active.long())[used_inv[point_is_active]]
        first_child = len(tree.parents) + 2**D*torch.arange(len(active), device=active.device)
        point_offset = ((points[point_is_active] >= tree.centers[active][active_inv])*bases).sum(-1)
        child_node = first_child[active_inv] + point_offset
        indices[point_is_active] = child_node

        trailing_ones = (slice(None),) + (None,)*D
        tree.children[active] = first_child[trailing_ones] + (subscript_offsets*bases).sum(-1)

        centers = tree.centers[active][trailing_ones] + center_offsets.float()/2**depth
        childtypes = center_offsets[None].expand_as(centers)

        n_children = len(active)*2**D
        children = arrdict.arrdict(
            parents=active.repeat_interleave(2**D),
            depths=tree.depths.new_full((n_children,), depth),
            centers=centers.reshape(-1, D),
            childtypes=childtypes.reshape(-1, D),
            terminal=tree.terminal.new_ones((n_children,)),
            children=tree.children.new_full((n_children,) + (2,)*D, -1))
        tree = arrdict.cat([tree, children])

    return tree, indices

def reflect(childtypes, directions):
    """Args:
        direction: [-1, 0, +1]^D
        childtypes: [-1, +1]^D
    """
    reflector = 1 - 2*directions.abs()
    return childtypes*reflector
    
def children(tree, indices, childtypes):
    subscripts = (childtypes + 1)/2
    return tree.children[(indices, *subscripts.T)]

def adj_neighbour(tree, indices, directions):
    if len(indices) == 0:
        return indices

    indices = torch.as_tensor(indices, dtype=tree.parents.dtype, device=tree.parents.device)
    directions = torch.as_tensor(directions, dtype=tree.parents.dtype, device=tree.parents.device)
    directions = directions[None].repeat_interleave(len(indices), 0) if directions.ndim == 1 else directions
    assert len(directions) == len(indices), 'There should be as many directions as indices'

    parents = tree.parents[indices]
    isnt_root = (parents >= 0)
    not_opposite = (tree.childtypes[indices] == directions).any(-1)
    recurse = isnt_root & not_opposite

    next_up = tree.parents[indices]
    next_up[recurse] = adj_neighbour(tree, parents[recurse], directions[recurse])

    next_up_is_interior = (next_up >= 0) & ~tree.terminal[next_up]
    reflected = reflect(tree.childtypes[indices][next_up_is_interior], directions[next_up_is_interior])
    next_down = next_up
    next_down[next_up_is_interior] = children(tree, next_up[next_up_is_interior], reflected)

    return next_down

def common_side(directions, childtypes):
    return directions*(directions == childtypes).type_as(directions)
    
def corner_neighbour(tree, indices, directions):
    if len(indices) == 0:
        return indices

    indices = torch.as_tensor(indices, dtype=tree.parents.dtype, device=tree.parents.device)
    directions = torch.as_tensor(directions, dtype=tree.parents.dtype, device=tree.parents.device)
    directions = directions[None].repeat_interleave(len(indices), 0) if directions.ndim == 1 else directions
    assert len(directions) == len(indices), 'There should be as many directions as indices'

    parents = tree.parents[indices]
    isnt_root = (parents >= 0)
    not_opposite = (tree.childtypes[indices] == directions).any(-1)
    recurse = isnt_root & not_opposite

    recurse_corner = (tree.childtypes[indices] == directions).all(-1) & recurse
    recurse_adj = ~recurse_corner & recurse
    adj_sides = common_side(directions, tree.childtypes[indices[recurse_adj]])
    next_up = tree.parents[indices]
    next_up[recurse_corner] = corner_neighbour(tree, parents[recurse_corner], directions)
    next_up[recurse_adj] = adj_neighbour(tree, parents[recurse_adj], adj_sides)

    next_up_is_interior = (next_up >= 0) & ~tree.terminal[next_up]
    reflected = -1*tree.childtypes[indices[next_up_is_interior]]
    next_down = next_up
    next_down[next_up_is_interior] = children(tree, next_up[next_up_is_interior], reflected)

    return next_down

def plot_tree(tree, ax=None, color=None):
    tree = tree.cpu().numpy()

    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.set_xlim(-1.1, +1.1)
    ax.set_ylim(-1.1, +1.1)
    ax.set_aspect(1)

    kwargs = {'color': color, 'fill': True, 'alpha': .25} if color else {'color': 'k', 'fill': False}
    for depth in np.unique(tree.depths):
        level = tree[tree.depths == depth]

        width = 2/2**depth
        corners = level.centers - np.array([1, 1])*width/2

        for corner in corners:
            ax.add_artist(mpl.patches.Rectangle(corner, width, width, **kwargs))
            
    return ax

def plot_problem(prob, q=.01, ax=None):
    prob = prob.cpu().numpy()
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.set_aspect(1)

    ax.scatter(*prob.targets.T, color='C0', label='targets', marker='.')

    charges = (prob.charges - prob.charges.min())/(prob.charges.max() - prob.charges.min() + 1e-6)
    ax.scatter(*prob.sources.T, color='red', s=10 + 100*charges, label='sources', marker='x')

    return ax

def run():
    torch.random.manual_seed(1)
    prob = aljpy.dotdict(
        sources=torch.tensor([[-.4, .4], [-.8, .8]]),
        charges=torch.tensor([1., 1.]),
        targets=torch.empty((0, 2)))

    prob = test.random_problem(T=30)

    scaled = scale(prob)
    cutoff = 2

    tree, indices = tree_indices(scaled, cutoff)

    ax = plot_problem(scaled)
    ax = plot_tree(tree, ax=ax)