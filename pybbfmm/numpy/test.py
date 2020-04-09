import aljpy
import numpy as np

def quad_kernel(a, b):
    return 1/((a - b)**2).sum(-1)

def random_problem(S=3, T=5, D=2):
    return aljpy.dotdict(
        sources=np.random.uniform(-1., +1., (S, D)),
        charges=np.random.uniform(.1, 1., (S,)),
        targets=np.random.uniform(-1., +1., (T, D)))

def solve(prob, kernel=quad_kernel):
    k = kernel(prob.targets[:, None], prob.sources[None, :])
    return (k*prob.charges).sum(-1)