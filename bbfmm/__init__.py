import aljpy
import numpy as np
import matplotlib.pyplot as plt
from . import test, chebyshev
from itertools import product

EPS = 1e-2

def limits(prob):
    points = np.concatenate([prob.sources, prob.targets])
    return np.stack([points.min(0) - EPS, points.max(0) + EPS])

def subdivide(xs, lims, cutoff=5):
    scaled = (xs - lims[0])/(lims[1] - lims[0])

    D = lims.shape[1]
    paths = np.zeros((len(xs), D), dtype=int)

    #TODO: You can probably get very smart about this and just cast the xs
    # to ints and look at their binary representation.
    for depth in range(32):
        occupancy = np.zeros((2**depth,)*D)
        np.add.at(occupancy, tuple(paths.T), 1)
        if occupancy.max() <= cutoff:
            break

        boundaries = (1/2 + np.arange(2**depth))/2**depth
        paths = 2*paths + (scaled >= boundaries[paths]).astype(int)
    else:
        raise ValueError('Paths seem very long')

    return paths, depth



def run():
    prob = test.random_problem(S=4, D=1)

    lims = limits(prob)

    source_paths = subdivide(prob.sources, lims, cutoff=1)
