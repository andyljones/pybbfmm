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

class Node:

    def __init__(self, parent, children, center):
        self.parent = parent
        self.children = children
        self.center = center

    def plot(self, ax=None):
        ax = plt.subplots()[1] if ax is None else ax
        if self.center is not None:
            ax.scatter(*self.center)

        for child in self.children.flatten():
            child.plot(ax)

class Leaf:

    def __init__(self, parent, points):
        self.parent = parent
        self.points = points

    def plot(self, ax):
        ax.scatter(*self.points.T, marker='.')


def tree(points, cutoff=5, parent=None):
    if len(points) > cutoff:
        center = points.min(0) + (points.max(0) - points.min(0))/2 

        node = Node(
            parent=parent, 
            children=np.empty((2,)*points.ndim, dtype=object),
            center=center)
        
        sides = (points > center)
        options = np.stack(list(product([False, True], repeat=points.ndim)))
        for option in options:
            child = tree(points[(sides == option).all(-1)], cutoff, node)
            node.children[tuple(option.astype(int))] = child
        
    else:
        node = Leaf(parent=parent, points=points)
    
    return node

