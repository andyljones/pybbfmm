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