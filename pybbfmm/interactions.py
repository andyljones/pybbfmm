import torch
from . import chebyshev
from aljpy import arrdict

def children(tree, indices, descent):
    subscripts = (descent + 1)/2
    return tree.children[(indices, *subscripts.T)]

def neighbours(tree, indices, directions):
    indices = torch.as_tensor(indices, dtype=tree.parents.dtype, device=tree.parents.device)
    directions = torch.as_tensor(directions, dtype=tree.parents.dtype, device=tree.parents.device)
    directions = directions[None].repeat_interleave(len(indices), 0) if directions.ndim == 1 else directions
    assert len(directions) == len(indices), 'There should be as many directions as indices'

    current = indices.clone()
    alive = [torch.ones_like(indices, dtype=torch.bool)]
    neighbour_descents = []
    while alive[-1].any():
        live = alive[-1] & (directions != 0).any(-1) & (current >= 0)
        alive.append(live)

        descent = tree.descent[current]
        neighbour_descents.append(descent*(1 - 2*directions.abs()))

        directions = (descent + directions).div(2).long() 
        current[live] = tree.parents[current[live]]

    for descent, live in zip(neighbour_descents[::-1], alive[::-1]):
        internal = ~tree.terminal[current] & (current >= 0) & live
        current[internal] = children(tree, current[internal], descent[internal])

    return current

def u_list(tree):
    """For childless nodes, the neighbouring childless nodes"""
    D = tree.children.ndim-1
    bs = tree.terminal.nonzero().squeeze(1)
    directions = chebyshev.flat_cartesian_product(torch.tensor([-1, 0, +1], device=bs.device), D)
    pairs = torch.cat([torch.stack([bs, neighbours(tree, bs, d)], -1) for d in directions])
    pairs = pairs[(pairs >= 0).all(-1) & tree.terminal[pairs[:, 1]]]

    partner_is_larger = tree.depths[pairs[:, 0]] > tree.depths[pairs[:, 1]]
    smaller_partners = torch.flip(pairs[partner_is_larger], (1,))
    pairs = torch.cat([pairs, smaller_partners])
    return pairs

def v_list(tree):
    """Children of the parent's colleagues that are separated from the node"""
    #TODO: Improve the memory consumption of this thing.
    #TODO: Factor out the neighbours list comp, that's reused by all the lists
    D = tree.children.ndim-1
    bs = tree.id
    parents = tree.parents[bs]
    directions = chebyshev.flat_cartesian_product(torch.tensor([-1, 0, +1], device=bs.device), D)
    directions = directions[(directions != 0).any(-1)]
    colleagues = torch.stack([neighbours(tree, parents, d) for d in directions], -1)

    friends_descents = chebyshev.flat_cartesian_product(torch.tensor([-1, +1], device=bs.device), D)
    friends = torch.stack([children(tree, colleagues, d) for d in friends_descents], -1)

    own_descents = tree.descent[bs]
    vector = -own_descents[:, None, None] + 4*directions[None, :, None] + friends_descents[None, None, :]
    friends[(vector.abs() <= 2).all(-1)] = -1

    pairs = torch.stack([bs[:, None, None].expand_as(friends), friends], -1)
    pairs = pairs[friends != -1]

    return pairs

def w_list(tree):
    """For childless nodes, descendents of colleagues whose parents are adjacent but
    which aren't themselves"""
    D = tree.children.ndim-1
    bs = tree.terminal.nonzero().squeeze(1)
    ds = chebyshev.flat_cartesian_product(torch.tensor([-1, 0, +1], device=bs.device), D)

    origins, colleagues, directions = [], [], []
    for d in ds:
        ns = neighbours(tree, bs, d)
        is_colleague = (tree.depths[bs] == tree.depths[ns])
        valid = is_colleague & ~tree.terminal[ns]
        origins.append(bs[valid])
        colleagues.append(ns[valid])
        directions.append(d[None].repeat_interleave(valid.sum(), 0))
    origins, colleagues, directions = torch.cat(origins), torch.cat(colleagues), torch.cat(directions, 0)

    ws = []
    parents = colleagues
    while parents.nelement():
        friends = tree.children[parents].reshape(-1, 2**D)
        distant = (tree.descent[friends] == directions[:, None, :]).any(-1)
        
        pairs = torch.stack([origins[:, None].expand_as(friends), friends], -1)
        ws.append(pairs[distant])
        
        mask = ~distant & ~tree.terminal[friends]
        origins, parents = pairs[mask].T
        directions = directions[:, None].repeat_interleave(2**D, 1)[mask]
    ws = torch.cat(ws)

    return ws

def lists(tree):
    lists = arrdict.arrdict(
                        u=u_list(tree),
                        v=v_list(tree),
                        w=w_list(tree))
    lists['x'] = torch.flip(lists['w'], (1,))
    return lists

