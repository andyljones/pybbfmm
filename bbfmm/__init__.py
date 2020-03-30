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

class Vertex:

    def __init__(self, lims):
        super().__init__()
        self.lims = lims

    def center(self):
        return self.lims.mean(0)

    def scale(self):
        return (self.lims[1] - self.lims[0])/2

    def into(self, xs):
        return (xs - self.center())/self.scale()

    def outof(self, xs):
        return xs*self.scale() + self.center()

    def nodes(self):
        D = self.lims.shape[-1]
        ms = np.arange(N)
        onedim = np.cos((ms+1/2)*np.pi/N)
        return np.stack(np.meshgrid(*[onedim]*D), axis=-1).reshape(-1, D)

class SourceInternal(Vertex):

    def __init__(self, children, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = children

    def height(self):
        return max(self.children)

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