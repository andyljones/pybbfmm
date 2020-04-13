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

def tree_nodes(scaled, cutoff=5):
    #TODO: Well this is a travesty of incomprehensibility. Verify it then explain yourself.
    D = scaled.sources.shape[1]
    points = torch.cat([scaled.sources, scaled.targets])
    leaves = points.new_zeros((len(points),), dtype=torch.long)

    tree = arrdict.arrdict(
        parents=leaves.new_full((1,), 0),
        depths=leaves.new_zeros((1,)),
        centers=points.new_zeros((1, D)),
        terminal=leaves.new_ones((1,), dtype=torch.bool),
        children=leaves.new_full((1,) + (2,)*D, -1))

    bases = 2**torch.arange(D, device=leaves.device)
    subscript_offsets = chebyshev.cartesian_product(torch.tensor([0, 1], device=leaves.device), D)
    center_offsets = chebyshev.cartesian_product(torch.tensor([-1., +1.], device=leaves.device), D)

    depth = 0
    while True:
        nodes, inv, counts = torch.unique(leaves, return_inverse=True, return_counts=True)
        tree.terminal[nodes] = (counts <= cutoff)
        
        node_active = ~tree.terminal[nodes]
        point_active = node_active[inv]
        if not point_active.any():
            break

        depth += 1
        
        parents = nodes[node_active]
        zeroth_child = len(tree.parents) + 2**D*torch.arange(len(parents), device=parents.device)
        point_offset = ((points[point_active] >= tree.centers[parents][inv[point_active]])*bases).sum(-1)
        child = zeroth_child + point_offset
        leaves[point_active] = child

        tree.children[parents] = zeroth_child[:, None] + (subscript_offsets*bases).sum(-1)

        centers = tree.centers[parents][:, None] + center_offsets/2**depth
        centers = centers.reshape(-1, D)

        children = arrdict.arrdict(
            parents=parents.repeat_interleave(2**D),
            depths=tree.depths.new_full((len(centers),), depth),
            centers=centers,
            terminal=tree.terminal.new_ones((len(centers),)),
            children=tree.children.new_full((len(centers),) + (2,)*D, -1))
        tree = arrdict.cat([tree, children])

    return tree, leaves


