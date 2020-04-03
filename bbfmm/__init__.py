"""
TODO: 
    * Generalize tree to cover both source and target nodes.
        * Subject to initial limits, it's the same centers time. Target tree can just nab 
    * Figure out interaction lists
    * 
"""
import aljpy
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def quad_kernel(a, b):
    return 1/((a - b)**2).sum(-1)

def random_problem(S=3, T=5, D=2):
    return aljpy.dotdict(
        sources=np.random.uniform(-1., +1., (S, D)),
        charges=np.random.uniform(.1, 1., (S,)),
        targets=np.random.uniform(-1., +1., (T, D)))

def analytic_solution(prob, kernel=quad_kernel):
    k = kernel(prob.sources[..., None, :, :], prob.targets[..., :, None, :])
    return (prob.charges[..., None, :]*k).sum(-1)

def plot(prob, soln=None, q=.01):
    fig, ax = plt.subplots()
    if soln is None:
        ax.scatter(*prob.targets.T, color='C0', label='targets', marker='.')
    else:
        lower, upper = np.quantile(soln, [q, 1-q]) 
        soln = (soln - lower)/(upper - lower)
        colors = plt.cm.viridis(soln)
        ax.scatter(*prob.targets.T, color=colors, marker='.')

    charges = (prob.charges - prob.charges.min())/(prob.charges.max() - prob.charges.min())
    ax.scatter(*prob.sources.T, color='red', s=10 + 100*charges, label='sources', marker='x')

    return ax

KERNEL = quad_kernel
N = 10 
EPS = 1e-2

def cartesian_product(xs, D):
    return np.stack(np.meshgrid(*([xs]*D), indexing='ij'), -1)

def flat_cartesian_product(xs, D):
    return cartesian_product(xs, D).reshape(-1, D)

def similarity(a, b):
    """
    Args:
        a: (m, d)
        b: (j, d)
    
    Returns:
        (m, j)
    """
    assert ((-1 <= a) & (a <= +1)).all()
    assert ((-1 <= b) & (b <= +1)).all()

    theta_a = np.arccos(a)[:, None, :, None]
    theta_b = np.arccos(b)[None, :, :, None]

    ks = np.arange(1, N)[None, None, None, :]
    terms = np.cos(ks*theta_a)*np.cos(ks*theta_b)
    return (1/N + 2/N*terms.sum(-1)).prod(-1)

def zero_weights(D):
    return 

def layer_grids(root):
    grids = [np.full([1]*root.dim(), root, dtype=object)]
    while True:
        blocks = np.vectorize(lambda v: v.pseudochildren())(grids[-1])
        grids.append(np.block(blocks.tolist()))
        if np.vectorize(lambda v: isinstance(v, Leaf))(grids[-1]).all():
            break
    return grids

def grid_neighbours(grid):
    D = grid.flatten()[0].dim()

    center = tuple([slice(1, -1)]*D)
    embedded = np.full(np.array(grid.shape)+2, Null(D), dtype=grid.dtype)
    embedded[center] = grid

    offsets = flat_cartesian_product(np.array([-1, 0, +1]), 2)
    offsets = offsets[~(offsets == (0, 0)).all(1)]
    center_indices = np.stack(np.indices(grid.shape) + 1, -1)
    neighbour_indices = center_indices[:, :, None, :] + offsets[None, None, :, :]
    neighbours = embedded[tuple(neighbour_indices[..., d] for d in range(D))] 

    return neighbours

def expand_parents(parents, D):
    W = parents.shape[0]
    indices = cartesian_product(np.arange(0, W, .5).astype(int), D)
    return parents[tuple(indices[..., d] for d in range(D))]

def interaction_sets(parents, children):
    D = children.flatten()[0].dim()
    null = Null(D)

    parents_neighbours= expand_parents(grid_neighbours(parents), D)
    child_nephews = np.block(np.vectorize(lambda v: v.pseudochildren().flatten())(parents_neighbours).tolist())

    child_neighbours = grid_neighbours(children)

    mask = (child_nephews[:, :, :, None] != child_neighbours[:, :, None, :]).all(-1)
    return np.where(mask, child_nephews, null), child_neighbours

class Vertex:

    def __init__(self, lims):
        super().__init__()
        self.lims = lims
    
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
        D = self.dim()
        ms = np.arange(N)
        onedim = np.cos((ms+1/2)*np.pi/N)
        return flat_cartesian_product(onedim, D)

class Internal(Vertex):

    def __init__(self, children, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = children

    def height(self):
        return max(c.height() for c in self.children.flatten()) + 1

    def pseudochildren(self):
        return self.children
    
class Leaf(Vertex):

    def __init__(self, points, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points = points

    def height(self):
        return 0

    def pseudochildren(self):
        return np.full((2,)*self.dim(), self)

class SourceInternal(Internal):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = np.zeros(N**self.dim())

    def weights(self):
        total = 0
        for child in self.children.flatten():
            nodes = self.into(child.outof(child.nodes()))
            total += (similarity(self.nodes(), nodes)*child.weights()).sum(-1)
        self.W = total
        return self.W

class SourceLeaf(Leaf):

    def __init__(self, charges, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.charges = charges
        self.W = np.zeros(N**self.dim())

    def weights(self):
        S = similarity(self.nodes(), self.into(self.points))
        self.W = (S*self.charges).sum(-1)
        return self.W

def assign_far_field(child, parent):
    g = np.zeros(N**child.dim())
    for ixn in child.interactions:
        if not isinstance(ixn, Null):
            K = KERNEL(child.outof(child.nodes())[:, None], ixn.outof(ixn.nodes())[None, :])
            g += (K*ixn.W).sum(-1)

    S = similarity(parent.into(child.outof(child.nodes())), parent.nodes())
    child.f = g + (S*parent.f).sum(-1)

class TargetInternal(Internal):

    def __init__(self, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = weights
        self.interactions = None
        self.f = np.zeros(N**self.dim())

    def far_field(self, parent):
        assign_far_field(self, parent)
        for child in self.children.flatten():
            child.far_field(self)
        
    def near_field(self):
        for child in self.children.flatten():
            child.near_field()
        
class TargetLeaf(Leaf):

    def __init__(self, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W = weights
        self.interactions = None
        self.neighbours = None

    def far_field(self, parent):
        assign_far_field(self, parent)

    def near_field(self):
        S = similarity(self.into(self.points), self.nodes())
        f = (S*self.f).sum(-1)

        for neighbour in self.neighbours.flatten():
            f += KERNEL(self.points, neighbour.sources)*neighbour.charges




class Null(Vertex):

    def __init__(self, D):
        super().__init__(lims=np.zeros((2, D)))

    def pseudochildren(self):
        return np.full((2,)*self.dim(), self)

    def __eq__(self, other):
        return isinstance(other, type(self))
    
    def __hash__(self):
        return 0


def allocate(points, lims):
    ds = np.arange(points.ndim)
    center = lims.mean(0)
    criticals = np.stack([lims[0], center, lims[1]])
    sides = (points > center)
    options = np.stack(list(product([False, True], repeat=points.ndim)))
    for option in options:
        mask = (sides == option).all(-1)
        option = option.astype(int)
        sublims = np.stack([criticals[option, ds], criticals[option+1, ds]])
        yield (tuple(option), mask, sublims)

def limits(sources, targets):
    points = np.concatenate([sources, targets])
    return np.stack([points.min(0) - EPS, points.max(0) + EPS])

def source_tree(points, charges, lims, cutoff=5):
    if len(points) > cutoff:
        children = np.empty((2,)*points.ndim, dtype=object)
        for option, mask, sublims in allocate(points, lims):
            children[option] = source_tree(points[mask], charges[mask], sublims, cutoff)
        return SourceInternal(children, lims)
    else:
        return SourceLeaf(charges, points, lims)

def target_tree(points, sources, lims, cutoff=5):
    if len(points) > cutoff:
        children = np.empty((2,)*points.ndim, dtype=object)
        source_children = sources.pseudochildren()
        for option, mask, sublims in allocate(points, lims):
            children[option] = target_tree(points[mask], source_children[option], sublims, cutoff)
        return TargetInternal(sources.W, children, lims)
    else:
        return TargetLeaf(sources.W, points, lims)


def assign_interaction_lists(root):
    grids = layer_grids(root)

    root.interactions = np.full_like(grids[0], Null(root.dim()))
    for (parents, children) in zip(grids, grids[1:]):
        sets, neighbours = interaction_sets(parents, children)
        sets = sets.reshape(-1, sets.shape[-1])
        neighbours = neighbours.reshape(-1, neighbours.shape[-1])
        for c, s in zip(children.flatten(), sets, neighbours):
            c.interactions = s
            if isinstance(c, Leaf):
                c.neighbours = neighbours

def test_similarity():
    lims = np.array([[-1], [+1]])
    v = Vertex(lims)

    def g(x):
        return (x**5).sum(-1)

    ns = v.nodes()
    xs = np.linspace(-1, +1, 101)[:, None]
    ghat = (similarity(xs, ns)*g(ns)).sum(-1)

    plt.plot(xs[:, 0], g(xs))
    plt.plot(xs[:, 0], ghat)

def run():
    prob = random_problem(S=100)

    lims = limits(prob.sources, prob.targets)
    stree = source_tree(prob.sources, prob.charges, lims)
    stree.weights()

    ttree = target_tree(prob.targets, stree, lims)
    assign_interaction_lists(ttree)

    for child in ttree.children:
        child.far_field(ttree)