import aljpy
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from . import chebyshev, test

KERNEL = test.quad_kernel
EPS = 1e-2

def layer_grids(root):
    grids = [np.full([1]*root.dim(), root, dtype=object)]
    while True:
        if np.vectorize(lambda v: isinstance(v, Leaf))(grids[-1]).all():
            break
        blocks = np.vectorize(lambda v: v.pseudochildren())(grids[-1])
        grids.append(np.block(blocks.tolist()))
    return grids

def grid_neighbours(grid):
    cheb = grid.flatten()[0].cheb
    D = cheb.D

    center = tuple([slice(1, -1)]*D)
    embedded = np.full(np.array(grid.shape)+2, Null(cheb), dtype=grid.dtype)
    embedded[center] = grid

    offsets = chebyshev.flat_cartesian_product(np.array([-1, 0, +1]), D)
    offsets = offsets[~(offsets == 0).all(1)]
    center_indices = np.stack(np.indices(grid.shape) + 1, -1)
    neighbour_indices = center_indices[..., None, :] + offsets
    neighbours = embedded[tuple(neighbour_indices[..., d] for d in range(D))] 

    return neighbours

def expand_parents(parents, D):
    W = parents.shape[0]
    indices = chebyshev.cartesian_product(np.arange(0, W, .5).astype(int), D)
    return parents[tuple(indices[..., d] for d in range(D))]

def interaction_sets(parents, children):
    cheb = parents.flatten()[0].cheb
    null = Null(cheb)

    parents_neighbours= expand_parents(grid_neighbours(parents), cheb.D)
    child_nephews = np.block(np.vectorize(lambda v: v.pseudochildren().flatten())(parents_neighbours).tolist())

    child_neighbours = grid_neighbours(children)

    mask = (child_nephews[..., :, None] != child_neighbours[..., None, :]).all(-1)
    return np.where(mask, child_nephews, null), child_neighbours

class Vertex:

    def __init__(self, lims, cheb):
        super().__init__()
        self.lims = lims
        self.cheb = cheb
    
    def dim(self):
        return self.lims.shape[-1]

    def center(self):
        return self.lims.mean(0)

    def scale(self):
        return (self.lims[1] - self.lims[0])/2

    def into(self, xs):
        return (xs - self.center())/self.scale()

    def outof(self, xs):
        return xs*self.scale() + self.center()

    def nodes(self):
        return self.cheb.nodes

    def __repr__(self):
        return f'{type(self).__name__}({str(hash(self))[-3:]})'

    def __str__(self):
        return repr(self)

class Internal(Vertex):

    def __init__(self, children, masks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = children
        self.masks = masks

    def height(self):
        return max(c.height() for c in self.children.flatten()) + 1

    def pseudochildren(self):
        return self.children

    def set_weights(self):
        total = 0
        for child in self.children.flatten():
            child.set_weights()
            nodes = self.into(child.outof(child.nodes()))
            total += self.cheb.anterpolate(nodes, child.W)
        self.W = total

    def set_far_field(self, parent=None):
        set_far_field(self, parent)
        for child in self.children.flatten():
            child.set_far_field(self)
        
    def values(self):
        V = np.zeros(len(self.masks.flatten()[0].targets))
        for child, mask in zip(self.children.flatten(), self.masks.flatten()):
            V[mask.targets] = child.values()
        return V
         
class Leaf(Vertex):

    def __init__(self, prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sources = prob.sources
        self.targets = prob.targets
        self.charges = prob.charges

    def height(self):
        return 0

    def pseudochildren(self):
        return np.full((2,)*self.dim(), self)

    def set_weights(self):
        self.W = self.cheb.anterpolate(self.into(self.sources), self.charges)

    def set_far_field(self, parent=None):
        set_far_field(self, parent)

    def values(self):
        V = self.cheb.interpolate(self.into(self.targets), self.f)

        for neighbour in self.neighbours.flatten():
            V += (KERNEL(self.targets[:, None], neighbour.sources[None, :])*neighbour.charges).sum(-1)
        
        V += (KERNEL(self.targets[:, None], self.sources[None, :])*self.charges).sum(-1)

        return V

class Null(Vertex):

    def __init__(self, cheb):
        super().__init__(np.zeros((2, cheb.D)), cheb)

    def pseudochildren(self):
        return np.full((2,)*self.dim(), self)

    def __eq__(self, other):
        return isinstance(other, type(self))
    
    def __hash__(self):
        return 0

    def __repr__(self):
        return f'{type(self).__name__}(000)'

def set_far_field(child, parent=None):
    g = child.cheb.zeros() 
    for ixn in child.interactions:
        K = KERNEL(child.outof(child.nodes())[:, None], ixn.outof(ixn.nodes())[None, :])
        g += (K*ixn.W).sum(-1)

    if parent is None:
        child.f = child.cheb.zeros()
    else:
        child.f = g + parent.cheb.interpolate(parent.into(child.outof(child.nodes())), parent.f)

def limits(prob):
    points = np.concatenate([prob.sources, prob.targets])
    return np.stack([points.min(0) - EPS, points.max(0) + EPS])

def subdivide(prob, lims):
    ds = np.arange(prob.sources.shape[-1])
    center = lims.mean(0)
    boundaries = np.stack([lims[0], center, lims[1]])
    options = np.stack(list(product([False, True], repeat=len(ds))))
    for option in options:
        masks = aljpy.dotdict(
            sources=((prob.sources >= center) == option).all(-1),
            targets=((prob.targets >= center) == option).all(-1))
        option = option.astype(int)
        sublims = np.stack([boundaries[option, ds], boundaries[option+1, ds]])
        yield (tuple(option), masks, sublims)

def required_depth(prob, lims, cutoff):
    if (len(prob.sources) > cutoff) or (len(prob.targets) > cutoff):
        depth = []
        for option, submasks, sublims in subdivide(prob, lims):
            subprob = aljpy.dotdict(
                sources=prob.sources[submasks.sources],
                targets=prob.targets[submasks.targets])
            depth.append(required_depth(subprob, sublims, cutoff))
        return max(depth) + 1
    else:
        return 0


def build_tree(prob, lims=None, cutoff=5, depth=None, cheb=None):
    D = prob.sources.shape[-1]
    lims = limits(prob) if lims is None else lims
    depth = required_depth(prob, lims, cutoff=cutoff) if depth is None else depth
    cheb = chebyshev.Chebyshev(10, D) if cheb is None else cheb
    if depth > 0:
        children = np.empty((2,)*D, dtype=object)
        masks = np.empty((2,)*D, dtype=object)
        for option, submasks, sublims in subdivide(prob, lims):
            subprob = aljpy.dotdict(
                sources=prob.sources[submasks.sources],
                targets=prob.targets[submasks.targets],
                charges=prob.charges[submasks.sources])
            masks[option] = submasks
            children[option] = build_tree(subprob, sublims, cutoff, depth-1, cheb)
        return Internal(children, masks, lims, cheb)
    else:
        return Leaf(prob, lims, cheb)

def set_interactions(root):
    grids = layer_grids(root)

    null = Null(root.cheb)
    root.interactions = np.empty((0,), dtype=object)
    if isinstance(root, Leaf):
        root.neighbours = np.empty((0,), dtype=object)
    for parents, children in zip(grids, grids[1:]):
        sets, neighbours = interaction_sets(parents, children)
        sets = sets.reshape(-1, sets.shape[-1])
        neighbours = neighbours.reshape(-1, neighbours.shape[-1])
        for c, s, n in zip(children.flatten(), sets, neighbours):
            c.interactions = np.array(list(set(s[(s != null) & (s != c)])))
            if isinstance(c, Leaf):
                c.neighbours = np.array(list(set(n[(n != null) & (n != c)])))

def solve(prob):
    root = build_tree(prob)
    root.set_weights()
    set_interactions(root)
    root.set_far_field()
    return root.values()


def run():
    prob = test.random_problem(S=50, T=50, D=2)

    vhat = solve(prob)

    v = test.solution(prob)
    np.around(vhat - v, 3)

# if W.dtype == object:
#     return np.array([interactions(w, scaled, cheb) for w in W])
# if W.shape[0] == 1:
#     return np.zeros_like(W)

# D, N = cheb.D, cheb.N
# width = W.shape[0]

# vectors, is_neighbour = nephew_vectors(cheb)
# vectors = (scaled.limits[1] - scaled.limits[0])/width*vectors

# nephew_kernel = KERNEL(np.zeros_like(vectors), vectors)
# interaction_kernel = np.where(is_neighbour, 0, nephew_kernel)

# mirrored = interaction_kernel[(slice(None, None, -1), slice(None))*D]

# W_dims = (width//2+2, 2)*D + (N**D,) + (1,)*D + (1,)
# Wp = np.pad(W, (((2, 2),)*D + ((0, 0),)))
# Wp = Wp.reshape(W_dims)

# ixns = scipy.signal.fftconvolve(
#     Wp,
#     mirrored,
#     mode='valid',
#     axes=np.arange(0, 2*D, 2)
# ).sum(tuple(2*i+1 for i in range(D)) + (2*D,), keepdims=True)

# squeeze = tuple(2*d+1 for d in range(D)) + (2*D,)
# ixns = ixns.squeeze(squeeze)

# axes = sum([(i, D+i) for i in range(D)], ()) + (2*D,)
# ixns = ixns.transpose(axes)

# ixns = ixns.reshape((width,)*D + (N**D,))

# def nephew_vectors(cheb):
#     # (nephew offset) + (nephew node) + (child offset) + (child node) + (D,) 
#     # (3, 2)*D + (N**D,) + (2,)*D + (N**D,) + (D,)
#     D, N = cheb.D, cheb.N

#     child_dims = (1, 1)*D + (1,) + (2,)*D + (1,) + (D,)
#     child_offsets = chebyshev.cartesian_product([0, 1], D).reshape(child_dims)

#     child_node_dims = (1, 1)*D + (1,) + (1,)*D + (N**D,) + (D,)
#     child_nodes = (cheb.nodes/2 + 1/2).reshape(child_node_dims)

#     nephew_dims = (3, 2)*D + (1,) + (1,)*D + (1,) + (D,)
#     nephew_offsets = chebyshev.cartesian_product(np.arange(-2, 4), D).reshape(nephew_dims)

#     nephew_node_dims = (1, 1)*D + (N**D,) + (1,)*D + (1,) + (D,)
#     nephew_nodes = (cheb.nodes/2 + 1/2).reshape(nephew_node_dims)

#     vectors = ((nephew_offsets + nephew_nodes) - (child_offsets + child_nodes))
#     is_neighbour = (abs(nephew_offsets - child_offsets) <= 1).all(-1)

#     return vectors, is_neighbour