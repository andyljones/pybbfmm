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