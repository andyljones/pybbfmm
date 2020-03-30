import aljpy
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def quad_kernel(a, b):
    return 1/((a - b)**2).sum(-1)

def random_problem(M=3, N=5, D=2):
    return aljpy.dotdict(
        sources=np.random.uniform(-1., +1., (M, D)),
        charges=np.random.uniform(.1, 1., (M,)),
        targets=np.random.uniform(-1., +1., (N, D)))

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
TERMS = 20

def chebyshev_nodes(lower, upper):
    ms = np.arange(TERMS)
    nodes = np.cos((2*ms+1)*np.pi/(2*TERMS))

    mid = (upper + lower)/2
    half = (upper - lower)/2
    return half[None, :]*nodes[:, None] + mid

class Node:

    def __init__(self, lims):
        super().__init__()
        self.lims = lims
        self.nodes = chebyshev_nodes(lims[0], lims[1])

class SourceNode(Node):

    def __init__(self, children, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = children

class SourceLeaf(Node):

    def __init__(self, points, aux, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points = points
        self.aux = aux

    def anterpolate(self):
        pass

def allocate(points, lims):
    center = lims.mean(0)
    criticals = np.stack([lims[0], center, lims[1]])
    breakpoint()
    sides = (points > center)
    options = np.stack(list(product([False, True], repeat=points.ndim)))
    for option in options:
        mask = (sides == option).all(-1)
        option = option.astype(int)
        sublims = np.stack([criticals[option], criticals[option+1]])

        yield (tuple(option), mask, sublims)

def source_tree(points, charges, cutoff=5, lims=None):
    lims = np.stack([points.min(0), points.max(0)]) if lims is None else lims
    if len(points) > cutoff:
        children = np.empty((2,)*points.ndim, dtype=object)
        for option, mask, sublims in allocate(points, lims):
            children[option] = source_tree(points[mask], charges[mask], cutoff, sublims)
        return SourceNode(children, lims)
    else:
        return SourceLeaf(points, charges, lims)
