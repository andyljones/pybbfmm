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
    k = kernel(prob.targets[:, None], prob.sources[None, :])
    return (k*prob.charges).sum(-1)

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
        if np.vectorize(lambda v: isinstance(v, Leaf))(grids[-1]).all():
            break
        blocks = np.vectorize(lambda v: v.pseudochildren())(grids[-1])
        grids.append(np.block(blocks.tolist()))
    return grids

def grid_neighbours(grid):
    D = grid.flatten()[0].dim()

    center = tuple([slice(1, -1)]*D)
    embedded = np.full(np.array(grid.shape)+2, Null(D), dtype=grid.dtype)
    embedded[center] = grid

    offsets = flat_cartesian_product(np.array([-1, 0, +1]), D)
    offsets = offsets[~(offsets == 0).all(1)]
    center_indices = np.stack(np.indices(grid.shape) + 1, -1)
    neighbour_indices = center_indices[..., None, :] + offsets
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

    mask = (child_nephews[..., :, None] != child_neighbours[..., None, :]).all(-1)
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
            total += (similarity(self.nodes(), nodes)*child.W).sum(-1)
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
        S = similarity(self.nodes(), self.into(self.sources))
        self.W = (S*self.charges).sum(-1)

    def set_far_field(self, parent=None):
        set_far_field(self, parent)

    def values(self):
        S = similarity(self.into(self.targets), self.nodes())
        V = (S*self.f).sum(-1)

        # if len(self.targets) > 0 and self.targets[0, 0] == 1.:
        #     breakpoint()

        for neighbour in self.neighbours.flatten():
            V += (KERNEL(self.targets[:, None], neighbour.sources[None, :])*neighbour.charges).sum(-1)
        
        V += (KERNEL(self.targets[:, None], self.sources[None, :])*self.charges).sum(-1)

        return V

class Null(Vertex):

    def __init__(self, D):
        super().__init__(lims=np.zeros((2, D)))

    def pseudochildren(self):
        return np.full((2,)*self.dim(), self)

    def __eq__(self, other):
        return isinstance(other, type(self))
    
    def __hash__(self):
        return 0

    def __repr__(self):
        return f'{type(self).__name__}(000)'

def set_far_field(child, parent=None):
    g = np.zeros(N**child.dim())
    for ixn in child.interactions:
        K = KERNEL(child.outof(child.nodes())[:, None], ixn.outof(ixn.nodes())[None, :])
        g += (K*ixn.W).sum(-1)

    if parent is None:
        child.f = np.zeros(len(child.nodes()))
    else:
        S = similarity(parent.into(child.outof(child.nodes())), parent.nodes())
        child.f = g + (S*parent.f).sum(-1)

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
    D = prob.sources.shape[-1]
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


def build_tree(prob, lims=None, cutoff=5, depth=None):
    D = prob.sources.shape[-1]
    lims = limits(prob) if lims is None else lims
    depth = required_depth(prob, lims, cutoff=cutoff) if depth is None else depth
    if depth > 0:
        children = np.empty((2,)*D, dtype=object)
        masks = np.empty((2,)*D, dtype=object)
        for option, submasks, sublims in subdivide(prob, lims):
            subprob = aljpy.dotdict(
                sources=prob.sources[submasks.sources],
                targets=prob.targets[submasks.targets],
                charges=prob.charges[submasks.sources])
            masks[option] = submasks
            children[option] = build_tree(subprob, sublims, cutoff, depth-1)
        return Internal(children, masks, lims)
    else:
        return Leaf(prob, lims)

def set_interactions(root):
    grids = layer_grids(root)

    null = Null(root.dim())
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
    prob = random_problem(S=50, T=50, D=2)

    vhat = solve(prob)

    v = analytic_solution(prob)
    np.around(vhat - v, 3)
