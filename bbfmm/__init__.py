"""
[Didactic code](https://github.com/lbluque/fmm/blob/master/fmm.py)

Summarizing the didactic code:

  * You build a quadtree of some sort over the source points
  * You take each leaf cell of the quadtree and calculate a multipole expansion of the cell's points around the cell's center
  * You work up from the leaves, shifting the expansions of the children to the center of the parent cell and summing them 
  * You build a quadtre of some sort over the target points.
  * You work down from the root, shifting the parent's inner expansion to the center of the child cell, then summing it with the converted, shifted expansions of all the source cells in the interaction set. 
  * At the leaves, you evaluate the inner expansion at each point, then add the contributions from the sources not abstracted by the expansion. 

So the key operations are:

  * Tree-building
  * Creating the leaves' outer expansions
  * Shifting an outer expansion to a new center
  * Shifting an inner expansion to a new center
  * Converting an outer expansion to an inner expansion
  * Summing interactions with neighbours not covered by the expansion


"""
import aljpy
import numpy as np
import matplotlib.pyplot as plt

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
    charges = (prob.charges - prob.charges.min())/(prob.charges.max() - prob.charges.min())
    ax.scatter(*prob.sources.T, color='red', s=10 + 100*charges, label='sources', marker='x')
    if soln is None:
        ax.scatter(*prob.targets.T, color='C1', label='targets', marker='.')
    else:
        lower, upper = np.quantile(soln, [q, 1-q]) 
        soln = (soln - lower)/(upper - lower)
        colors = plt.cm.viridis(soln)
        ax.scatter(*prob.targets.T, color=colors, marker='.')
    return ax