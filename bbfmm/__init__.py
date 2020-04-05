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
    paths = np.stack(paths)

    depth, _, D = paths.shape
    bases = (2**D)**np.arange(depth-1, -1, -1)[:, None] * 2**np.arange(D-1, -1, -1)[None, :]
    indices = (bases[:, None, :]*paths).sum(2).sum(0)

    totals = np.zeros((2**D)**depth)
    np.add.at(totals, indices, 1)

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

    return np.stack(paths)