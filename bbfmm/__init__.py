import aljpy
import numpy as np
import matplotlib.pyplot as plt
from . import test, chebyshev
from itertools import product

EPS = 1e-2

def limits(prob):
    points = np.concatenate([prob.sources, prob.targets])
    return np.stack([points.min(0) - EPS, points.max(0) + EPS])

def occupancy(paths, d, D):
    occupancy = np.zeros((2**d,)*D)
    np.add.at(occupancy, tuple(paths.T), 1)
    return occupancy.max()

def tree_indices(prob, lims, cutoff=5):
    sources = (prob.sources - lims[0])/(lims[1] - lims[0])
    targets = (prob.targets - lims[0])/(lims[1] - lims[0])

    D = lims.shape[1]
    si = np.zeros((len(sources), D), dtype=int)
    ti = np.zeros((len(targets), D), dtype=int)

    #TODO: You can probably get very smart about this and just cast the xs
    # to ints and look at their binary representation.
    for d in range(32):
        s_done = occupancy(si, d, D) <= cutoff
        t_done = occupancy(ti, d, D) <= cutoff
        if s_done and t_done:
            break

        boundaries = (1/2 + np.arange(2**d))/2**d
        si = 2*si + (sources >= boundaries[si]).astype(int)
        ti = 2*ti + (targets >= boundaries[ti]).astype(int)
    else:
        raise ValueError('Paths seem very long')

    return aljpy.dotdict(
        si=si, 
        ti=ti, 
        depth=d)

def weights(prob, cheb, tree):
    pass


def run():
    prob = test.random_problem(S=4, D=1)

    lims = limits(prob)

    indices = tree_indices(prob, lims, cutoff=1)
