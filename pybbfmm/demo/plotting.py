import numpy as np
import matplotlib.pyplot as plt
from aljpy import recording
import scipy.ndimage

def viewport(charges, points, threshold=1e-1, eps=10):
    threshold = threshold*charges.max()
    lims = np.stack([
        points[charges > threshold].min(0) - eps,
        points[charges > threshold].max(0) + eps])
    center = lims.mean(0)
    scale = (lims[1] - lims[0]).max()/2
    return center, scale

def interpolated_viewport(charges, next, increment, points):
    c1, s1 = viewport(charges, points)
    c2, s2 = viewport(next, points)
    return c1*(1-increment) + c2*increment, s1*(1-increment) + s2*increment

def plot(charges, next, increment, points, threshold=1e-1, res=1000):
    center, scale = interpolated_viewport(charges, next, increment, points)

    visible = (points > center - scale).all(-1) & (points < center + scale).all(-1)

    points, charges = points[visible], charges[visible]

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.set_title(f'{(charges > threshold).sum()} infected')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.set_size_inches(res/fig.get_dpi(), res/fig.get_dpi())

    xy = (res*(points - (center - scale))/(2*scale))
    ij = np.stack([res - xy[:, 1], xy[:, 0]], -1).astype(int).clip(0, res-1)

    sums = np.zeros((res, res))
    np.add.at(sums, tuple(ij.T), charges)

    counts = np.zeros((res, res))
    np.add.at(counts, tuple(ij.T), np.ones_like(charges))

    means = np.full((res, res), 0.)
    np.divide(sums, counts, out=means, where=counts > 0)

    (l, b), (r, t) = center - scale, center + scale
    means = scipy.ndimage.gaussian_filter(means, .025/scale*res)
    ax.imshow(means, extent=(l, r, b, t))
    
    return fig

def animate(infected, points):
    smoothing = 4
    firsts = [i for i in infected[:-1] for _ in range(smoothing)]
    seconds = [i for i in infected[1:] for _ in range(smoothing)]
    increment = np.arange(0, len(infected)-1, 1/4) % 1
    encoder = recording.parallel_encode(render, firsts, seconds, increment, points=points, N=4, fps=4*smoothing)
    return recording.notebook(encoder)

# from pybbfmm.demo import *
# import pathlib
# import pickle

# globals().update(pickle.loads(pathlib.Path('output/tmp.pkl').read_bytes()))
