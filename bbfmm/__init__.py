import aljpy
import numpy as np
import matplotlib.pyplot as plt
from . import test, chebyshev
from itertools import product

EPS = 1e-2

def limits(prob):
    points = np.concatenate([prob.sources, prob.targets])
    return np.stack([points.min(0) - EPS, points.max(0) + EPS])

def subdivide(xs, lims):
    lims = np.repeat(lims[None], len(xs), 0)
    paths = []

    for _ in range(5):
        center = lims.mean(1)
        boundaries = np.stack([lims[:, 0], center, lims[:, 1]], 1)

        fork = (xs >= center).astype(int)

        lims = np.concatenate([
            np.take_along_axis(boundaries, fork[:, None]+0, axis=1),
            np.take_along_axis(boundaries, fork[:, None]+1, axis=1)], -2)

        paths.append(fork)
    paths = np.stack(paths)