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

def accumulate(indices, vals, length):
    totals = vals.new_zeros((length,) + vals.shape[1:])
    totals.index_add_(0, indices, vals)
    return totals.reshape(-1, vals.shape[1:])

def value_counts(indices, length):
    vals = indices.new_ones((len(indices),), dtype=torch.int32)
    return accumulate(indices, vals, length)

def children(indices, tree, active):
    pass

def tree_indices(scaled, cutoff=5):
    #TODO: Well this is a travesty of incomprehensibility. Verify it then explain yourself.
    D = scaled.sources.shape[1]

    points = torch.cat([scaled.sources, scaled.targets])
    indices = points.new_zeros((len(points),), dtype=torch.long)

    tree = arrdict.arrdict(
        parents=indices.new_full((1,), 0),
        depths=indices.new_zeros((1,)),
        centers=points.new_zeros((1, D)),
        terminal=indices.new_ones((1,), dtype=torch.bool),
        children=indices.new_full((1,) + (2,)*D, -1))

    bases = 2**torch.flip(torch.arange(D, device=indices.device), (0,))
    subscript_offsets = chebyshev.cartesian_product(torch.tensor([0, 1], device=indices.device), D)
    center_offsets = chebyshev.cartesian_product(torch.tensor([-1., +1.], device=indices.device), D)

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

        tree.children[active] = first_child[(slice(None),) + (None,)*D] + (subscript_offsets*bases).sum(-1)

        centers = tree.centers[active][(slice(None),) + (None,)*D] + center_offsets/2**depth
        centers = centers.reshape(-1, D)

        children = arrdict.arrdict(
            parents=active.repeat_interleave(2**D),
            depths=tree.depths.new_full((len(centers),), depth),
            centers=centers,
            terminal=tree.terminal.new_ones((len(centers),)),
            children=tree.children.new_full((len(centers),) + (2,)*D, -1))
        tree = arrdict.cat([tree, children])

    return tree, indices

def plot_tree(tree, ax=None):
    tree = tree.cpu().numpy()

    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.set_xlim(-1.1, +1.1)
    ax.set_ylim(-1.1, +1.1)
    ax.set_aspect(1)

    for depth in np.unique(tree.depths):
        level = tree[tree.depths == depth]

        width = 2/2**depth
        corners = level.centers - np.array([1, 1])*width/2

        for corner in corners:
            ax.add_artist(mpl.patches.Rectangle(corner, width, width, color='k', fill=False))
            
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