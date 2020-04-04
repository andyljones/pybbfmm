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

def limits(prob):
    points = np.concatenate([prob.sources, prob.targets])
    return np.stack([points.min(0) - EPS, points.max(0) + EPS])

def required_depth(xs, cutoff, lims=None):
    # Trick here is to map the limits to [0, 1], then bisect until the cutoffs are met
    if xs.ndim > 1:
        return max(required_depth(x, cutoff, l) for x, l in zip(xs.T, lims.T))
    if lims is not None:
        xs = (xs - lims[0])/(lims[1] - lims[0])

    if len(xs) > cutoff:
        return 1 + max(
                required_depth(2*xs[xs < .5], cutoff), 
                required_depth(2*xs[xs >= .5]-1, cutoff))
    else:
        return 0
    
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

