import numpy as np
import matplotlib.pyplot as plt
from aljpy import recording
from aljpy.plot import si_suffix
import scipy.ndimage
import scipy.interpolate
import pathlib
from IPython.display import display, HTML

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

def plot(charges, center, scale, step, points, threshold=1e-1, res=1000):
    visible = (points > center - scale).all(-1) & (points < center + scale).all(-1)

    points, charges = points[visible], charges[visible]

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    ax.set_title(f'#{step}: {si_suffix((charges > threshold).sum())} infected')
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
    means = scipy.ndimage.gaussian_filter(means, .1/scale*res)
    ax.imshow(means, extent=(l, r, b, t))
    
    return fig

def animate(infected, points, smoothing=4):
    print('Smoothing viewports...')
    centers, scales = interpolated_viewports(infected, points, smoothing)
    steps = (np.arange(len(centers))/smoothing).astype(int)
    repeated = [i for i in infected for _ in range(smoothing)]
    encoder = recording.parallel_encode(
                                plot, repeated, centers, scales, steps, 
                                points=points, N=0, fps=4*smoothing)

    html, _ = recording.html_tag(encoder)
    pathlib.Path('output/demo.html').write_text(html)

    return display(HTML(html))