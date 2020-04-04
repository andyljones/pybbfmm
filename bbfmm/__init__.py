import aljpy
import numpy as np
import matplotlib.pyplot as plt
from . import test, chebyshev
from itertools import product

EPS = 1e-2

class Grids:

    def __init__(self, depth, dim, tail=(), dtype=float):
        n = ((2**dim)**depth - 1)/((2**dim) - 1)
        self._storage = np.zeros((n, *tail), dtype=dtype)

def cartesian_product(xs, D):
    return np.stack(np.meshgrid(*([xs]*D), indexing='ij'), -1)

def flat_cartesian_product(xs, D):
    return cartesian_product(xs, D).reshape(-1, D)

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

def make_grids(prob, cutoff):
    lims = limits(prob)

    while True:
        finished = (len(prob.sources) <= cutoff) and (len(prob.targets) <= cutoff)
        if finished:
            pass