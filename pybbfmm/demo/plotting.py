import numpy as np
import matplotlib.pyplot as plt
from aljpy import recording
from aljpy.plot import si_suffix
import scipy.ndimage
import scipy.interpolate
import scipy as sp
import pathlib
from IPython.display import display, HTML
from matplotlib.colors import LinearSegmentedColormap

def viewport(charges, points, threshold=1e-1, eps=10):
    threshold = threshold*charges.max()
    lims = np.stack([
        points[charges > threshold].min(0) - eps,
        points[charges > threshold].max(0) + eps])
    center = lims.mean(0)
    scale = (lims[1] - lims[0]).max()/2
    return center, scale

def interpolated_viewports(infected, points, smoothing):
    centers, scales = zip(*[viewport(i, points) for i in infected])
    centers, scales = np.stack(centers), np.stack(scales)

    t0s = np.arange(len(centers))
    ts = np.linspace(0, len(centers), smoothing*len(centers)+1)[:-1]
    smooth_centers = np.stack([scipy.interpolate.UnivariateSpline(t0s, c, s=5000, ext='const')(ts) for c in centers.T], -1)
    smooth_scales = scipy.interpolate.UnivariateSpline(t0s, scales, s=5000, ext='const')(ts)

    return smooth_centers, smooth_scales

def plot(charges, center, scale, step, points, threshold=1e-1, res=2000):
    visible = (points > center - scale).all(-1) & (points < center + scale).all(-1)

    points, charges = points[visible], charges[visible]

    dpi = plt.gcf().get_dpi()
    fig = plt.figure(figsize=(res/dpi, res/dpi))
    ax = plt.Axes(fig, [0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    fig.add_axes(ax)
    ax.set_aspect(1)
    ax.set_title(f'#{step}: {si_suffix((charges > threshold).sum())} infected', y=.95, fontdict={'fontsize': 24})
    ax.set_xticks([])
    ax.set_yticks([])

    xy = (res*(points - (center - scale))/(2*scale))
    ij = np.stack([res - xy[:, 1], xy[:, 0]], -1).astype(int).clip(0, res-1)

    sums = np.zeros((res, res))
    np.add.at(sums, tuple(ij.T), charges)
    sums = scipy.ndimage.gaussian_filter(sums, .1/scale*res)

    counts = np.zeros((res, res))
    np.add.at(counts, tuple(ij.T), np.ones_like(charges))
    counts = scipy.ndimage.gaussian_filter(counts, .1/scale*res)

    means = np.full((res, res), 0.)
    np.divide(sums, counts, out=means, where=counts > 0)

    (l, b), (r, t) = center - scale, center + scale

    cmap = LinearSegmentedColormap.from_list('contagion', ['#F8F8F8', '#ff7f0e'])
    fig.patch.set_facecolor('#F8F8F8')
    vmax = 10*sp.stats.norm(0, .1/scale*res).pdf(0)
    ax.imshow(means, extent=(l, r, b, t), cmap=cmap, vmax=vmax)
    
    arr = recording.array(fig)
    plt.close(fig)

    return arr

def animate(infected, points, smoothing=4, N=0):
    print('Smoothing viewports...')
    centers, scales = interpolated_viewports(infected, points, smoothing)
    steps = (np.arange(len(centers))/smoothing).astype(int)
    repeated = [i for i in infected for _ in range(smoothing)]
    encoder = recording.parallel_encode(
                                plot, repeated, centers, scales, steps, 
                                points=points, N=N, fps=4*smoothing)

    pathlib.Path('output/demo.mp4').write_bytes(encoder.value)

    return display(HTML(recording.html_tag(encoder)[0]))