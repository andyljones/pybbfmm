import aljpy
from . import test, chebyshev, orthantree, ragged
import torch

EPS = 1e-2

def limits(prob):
    points = torch.cat([prob.sources, prob.targets])
    lims = []
    # This madness is because max(dim=0) is 2000x slower than max()
    # https://github.com/pytorch/pytorch/issues/36900
    for d in range(points.shape[1]):
        lims.append(torch.stack([points[:, d].min() - EPS, points[:, d].max() + EPS]))
    return torch.stack(lims, -1)

def scale(prob):
    lims = limits(prob)
    mid = (lims[0] + lims[1])/2
    scale = (lims[1] - lims[0])/2
    return aljpy.dotdict(
        limits=lims,
        scale=scale,
        sources=(prob.sources - mid)/scale,
        charges=prob.charges,
        targets=(prob.targets - mid)/scale,
        kernel=lambda a, b: prob.kernel(a*scale, b*scale))

def weights(scaled, cheb, tree, indices):
    leaves = tree.terminal.nonzero()
    box_to_source = ragged.invert_indices(indices.sources, len(tree.id))
    W = scaled.charges.new_zeros((len(tree.id), cheb.N**cheb.D))
    for k in range(box_to_source.max_k):
        sources, mask = box_to_source.kth(leaves, k)
        boxes = leaves[mask]
        loc = 2**tree.depths[boxes, None]*(scaled.sources[sources] - tree.centers[boxes])
        W[boxes] += cheb.similarity(loc, cheb.nodes)*scaled.charges[sources, None]

    coeffs = cheb.upwards_coeffs()
    dot_dims = (list(range(1, cheb.D+2)), list(range(1, cheb.D+2)))

    parents = tree.parents[indices.sources]
    while (parents >= 0).any():
        parents = parents[parents >= 0]
        W[parents] = torch.tensordot(W[tree.children[parents]], coeffs, dot_dims)
        parents = tree.parents[parents] 

    return W

def node_locations(cheb, tree, indices):
    return cheb.nodes[None]/2**tree.depths[indices, None, None] + tree.centers[indices, None, :]

def v_interactions(W, scaled, cheb, tree, scheme):
    ixns = torch.zeros_like(W) 
    for v in scheme.v:
        scale = 1/2**v.depth.float()
        boxes = scale*cheb.nodes
        friends = scale*(2*v.offset + cheb.nodes)
        k = scaled.kernel(boxes[None, :], friends[:, None])
        ixns[v.boxes] += W[v.friends] @ k
    
    return ixns

def x_interactions(scaled, cheb, tree, indices, scheme):
    box_to_source = ragged.invert_indices(indices.sources, len(tree.id))

    ixns = scaled.charges.new_zeros((len(tree.id), cheb.N**cheb.D))
    for p in range(scheme.x.max_k):
        partner, box_mask = scheme.x.kth(tree.id, p)
        for s in range(box_to_source.max_k):
            sources, partner_mask = box_to_source.kth(partner, s) 
            boxes = tree.id[box_mask][partner_mask]
            K = scaled.kernel(
                    node_locations(cheb, tree, boxes), 
                    scaled.sources[sources, None, :])
            ixns[boxes] += K*scaled.charges[sources, None]
    
    return ixns

def w_interactions(W, scaled, cheb, tree, indices, scheme):
    ixns = scaled.charges.new_zeros(len(scaled.targets))
    for p in range(scheme.w.max_k):
        partner, box_mask = scheme.w.kth(indices.targets, p)
        targets = scaled.targets[box_mask, None, :]
        partner_nodes = node_locations(cheb, tree, partner)
        K = scaled.kernel(targets, partner_nodes)
        ixns[box_mask.nonzero().squeeze(1)] += (K*W[partner]).sum(-1)
    return ixns

def u_interactions(scaled, indices, scheme):
    box_to_source = ragged.invert_indices(indices.sources, scheme.u.range)
    target_idxs = torch.arange(len(scaled.targets), device=scaled.targets.device)
    ixns = scaled.charges.new_zeros(len(scaled.targets))
    for b in range(scheme.u.max_k):
        boxes, target_mask = scheme.u.kth(indices.targets, b)
        for s in range(box_to_source.max_k):
            sources, box_mask = box_to_source.kth(boxes, s)
            targets = target_idxs[target_mask][box_mask]
            K = scaled.kernel(
                    scaled.targets[target_mask][box_mask], 
                    scaled.sources[sources])
            ixns[targets] += K*scaled.charges[sources]
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

def presolve(prob):
    cheb = chebyshev.Chebyshev(4, prob.sources.shape[1], device='cuda')
    scaled = scale(prob)
    tree, indices, depths = orthantree.orthantree(scaled)
    scheme = orthantree.interaction_scheme(tree, depths)
    return aljpy.dotdict(
        cheb=cheb, 
        scaled=scaled, 
        tree=tree, 
        scheme=scheme,
        indices=indices,
        depths=depths)

def evaluate(cheb, scaled, tree, scheme, indices, depths):

    W = weights(scaled, cheb, tree, indices)

    v = v_interactions(W, scaled, cheb, tree, scheme)
    x = x_interactions(scaled, cheb, tree, indices, scheme)

    F = far_field(W, v, x, cheb, tree)
    f = target_far_field(F, scaled, cheb, tree, indices)

    w = w_interactions(W, scaled, cheb, tree, indices, scheme)
    u = u_interactions(scaled, indices, scheme)

    potential = f + w + u
    return potential

def solve(prob):
    presoln = presolve(prob)
    return evaluate(**presoln)

def run():
    prob = test.random_problem(S=100, T=100)
    soln = solve(prob)
    ref = test.solve(prob)
    (soln - ref).pow(2).sum()/ref.pow(2).sum()

def mem_benchmark():
    from pytorch_memlab import LineProfiler 
    n = int(8e6)
    prob = test.random_problem(S=n, T=n)

    with LineProfiler(solve) as prof:
        solve(prob)
    
    prof.print_stats()