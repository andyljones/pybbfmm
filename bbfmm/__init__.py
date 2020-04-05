import aljpy
import numpy as np
import matplotlib.pyplot as plt
from . import test, chebyshev
from itertools import product

EPS = 1e-2

def limits(prob):
    points = np.concatenate([prob.sources, prob.targets])
    return np.stack([points.min(0) - EPS, points.max(0) + EPS])

def occupancy(paths):
    #TODO: numpy has a 32-dim limit hardcoded. So this might cause trouble
    # if there's a 32-length path, which isn't beyond belief in a non-uniform
    # input distribution
    paths = np.stack(paths, 1) if isinstance(paths, list) else paths
    _, depth, D = paths.shape
    totals = np.zeros((2,)*(D*depth), dtype=int)
    np.add.at(totals, tuple(paths.reshape(-1, D*depth).T), 1)

    return totals

def subdivide(xs, lims, cutoff=5):
    if len(xs) < cutoff:
        return np.empty((0, len(xs), 2))

    lims = np.repeat(lims[None], len(xs), 0)
    paths = []

    while True:
        center = lims.mean(1)
        boundaries = np.stack([lims[:, 0], center, lims[:, 1]], 1)

        fork = (xs >= center).astype(int)
        paths.append(fork)

        lims = np.concatenate([
            np.take_along_axis(boundaries, fork[:, None]+0, axis=1),
            np.take_along_axis(boundaries, fork[:, None]+1, axis=1)], -2)

        if occupancy(paths).max() <= cutoff:
            break

    return np.stack(paths, 1)

def sublimits(lims, depth):
    for d in range(depth):
        center = lims.mean(-2)
        lower = np.stack([lims[..., 0, :], center], -2)
        upper = np.stack([center, lims[..., 1, :]], -2)
        lims = np.stack([lower, upper], -3)
    return lims


def run():
    prob = test.random_problem(S=4, D=1)

    lims = limits(prob)

    source_paths = subdivide(prob.sources, lims, cutoff=1)
