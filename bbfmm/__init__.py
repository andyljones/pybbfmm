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

def cartesian_product(xs, D):
    return np.stack(np.meshgrid(*([xs]*D)), -1)

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
    center_indices = np.stack(list(np.indices(grid.shape) + 1), -1)
    neighbour_indices = center_indices[:, :, None, :] + offsets[None, None, :, :]
    neighbours = embedded[tuple(neighbour_indices[..., d] for d in range(D))] 

    return neighbours

def expand_parents(parents, D):
    W = parents.shape[0]
    indices = cartesian_product(np.arange(0, W, .5).astype(int), D)
    return parents[tuple(indices[..., d] for d in range(D))]

def interaction_sets(children, parents):
    D = children.flatten()[0].dim()
    null = Null(D)

    parents_neighbours = expand_parents(grid_neighbours(parents), D)
    child_nephews = np.block(np.vectorize(lambda v: v.pseudochildren().flatten())(parents_neighbours).tolist())

    child_neighbours = grid_neighbours(children)

    mask = (child_nephews[:, :, :, None] != child_neighbours[:, :, None, :]).all(-1)
    return np.where(mask, child_nephews, null)

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

class SourceInternal(Vertex):

    def __init__(self, children, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = children

    def height(self):
<<<<<<< HEAD
        return max(self.children)
=======
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
>>>>>>> fbab82c1... Adds grid generation

    def weights(self):
        total = 0
        for child in self.children.flatten():
            nodes = self.into(child.outof(child.nodes()))
            total += (similarity(self.nodes(), nodes)*child.weights()).sum(-1)
        return total

class SourceLeaf(Vertex):

    def __init__(self, points, charges, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points = points
        self.charges = charges

    def weights(self):
        S = similarity(self.nodes(), self.into(self.points))
        return (S*self.charges).sum(-1)

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

def source_tree(points, charges, cutoff=5, lims=None):
    lims = np.stack([points.min(0), points.max(0)]) if lims is None else lims
    if len(points) > cutoff:
        children = np.empty((2,)*points.ndim, dtype=object)
        for option, mask, sublims in allocate(points, lims):
            children[option] = source_tree(points[mask], charges[mask], cutoff, sublims)
        return SourceInternal(children, lims)
    else:
        return SourceLeaf(points, charges, lims)


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
    prob = random_problem(S=10)

    root = source_tree(prob.sources, prob.charges)

    ws = root.weights()

    ax = plot(prob)
    ax.scatter(*root.outof(root.nodes()).T, c=ws)