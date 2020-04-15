import aljpy
from aljpy import arrdict
from . import test, chebyshev, interactions
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

def orthant_tree(scaled, cutoff=5):
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
        descent=indices.new_zeros((1, D)))

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
        descent = center_offsets[None].expand_as(centers)

        n_children = len(active)*2**D
        children = arrdict.arrdict(
            parents=active.repeat_interleave(2**D),
            depths=tree.depths.new_full((n_children,), depth),
            centers=centers.reshape(-1, D),
            descent=descent.reshape(-1, D),
            terminal=tree.terminal.new_ones((n_children,)),
            children=tree.children.new_full((n_children,) + (2,)*D, -1))
        tree = arrdict.cat([tree, children])

    tree['id'] = torch.arange(len(tree.parents), device=points.device)

    return tree, indices

def plot_tree(tree, ax=None, color={}, number=False):
    tree = tree.cpu().numpy()

    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.set_xlim(-1.1, +1.1)
    ax.set_ylim(-1.1, +1.1)
    ax.set_aspect(1)

    color = {int(v): k for k, vs in color.items() for v in vs}

    for id, center, depth in zip(tree.id, tree.centers, tree.depths):
        if id in color:
            kwargs = {'color': color[id], 'fill': True, 'alpha': .25} 
        else:
            kwargs = {'color': 'k', 'fill': False, 'alpha': .25}
        if number:
            ax.annotate(str(id), center, ha='center', va='center')
        width = 2/2**depth
        corner = center - np.array([1, 1])*width/2
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

    prob = test.random_problem(T=500)

    scaled = scale(prob)

    tree, indices = orthant_tree(scaled)

    lists = interactions.lists(tree)

    b = int(tree.terminal.nonzero().squeeze(-1)[0])
    color = {'u': 'C0', 'v': 'C1', 'w': 'C2', 'x': 'C3', 'y': 'C4'}
    key = {k: mpl.patches.Rectangle((0., 0.), 1, 1, color=c, alpha=.25) for k, c in color.items()}
    color = {color[k]: l[l[:, 0] == b, 1] for k, l in lists.items()}
    color['C4'] = interactions.y_list(tree, b)
    ax = plot_tree(tree, color=color)
    ax.legend(key.values(), key.keys())