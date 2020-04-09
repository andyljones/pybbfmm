import aljpy
from jax import numpy as np
import jax
import numpy as onp

def quad_kernel(a, b):
    return 1/((a - b)**2).sum(-1)

def random_problem(S=3, T=5, D=2):
    return aljpy.dotdict(
        sources=onp.random.uniform(-1., +1., (S, D)),
        charges=onp.random.uniform(.1, 1., (S,)),
        targets=onp.random.uniform(-1., +1., (T, D))).map(np.asarray)

def solve(prob, kernel=quad_kernel):
    k = kernel(prob.targets[:, None], prob.sources[None, :])
    return (k*prob.charges).sum(-1)