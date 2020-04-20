import requests
import zipfile
import aljpy
import yaml
import numpy as np
from io import BytesIO
import torch
from aljpy import arrdict, recording
from . import adaptive
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage

LOGIN = 'https://data-package.ceh.ac.uk/sso/login'
DATA = 'https://data-package.ceh.ac.uk/data/0995e94d-6d42-40c1-8ed4-5090d82471e1.zip'

def credentials():
    return yaml.safe_load(open('credentials.yml', 'r'))['CEH']

@aljpy.autocache()
def _density_map():
    with requests.Session() as sess:
        r = sess.post(LOGIN, data={**credentials(), 'success': '/data/0995e94d-6d42-40c1-8ed4-5090d82471e1', 'failure': '/'})
        r.raise_for_status()
        r = sess.get(DATA)
        r.raise_for_status()
    return r.content

def density_map():
    """Fetches a 1km-resolution map of the UK's population density as a numpy array."""
    content = _density_map()
    with zipfile.ZipFile(BytesIO(content)) as zf:
        filename = '0995e94d-6d42-40c1-8ed4-5090d82471e1/data/UK_residential_population_2011_1_km.asc'
        asc = zf.read(filename)
    arr = np.loadtxt(BytesIO(asc), skiprows=6)
    arr[arr == -9999] = np.nan
    return arr

def pop_points(n=1e3):
    """Distribute the population around the country, returning a Px2 array of their coordinates relative to the 
    bottom-right corner."""
    density = density_map()
    density = density/np.nansum(density)
    density[np.isnan(density)] = 0.

    indices = np.random.choice(np.arange(density.size), (int(n),), p=density.flatten())
    subscripts = np.stack(np.unravel_index(indices, density.shape), -1)

    # Pick a uniformly random spot in the grid cell
    ij = subscripts + np.random.rand(len(subscripts), 2)
    xy = np.stack([ij[:, 1], len(density) - ij[:, 0]], -1)

    return xy

def risk_kernel(a, b):
    d = (a - b).pow(2).sum(-1).pow(.5)
    return .0001 * 1/(1 + (d/4)**3)

def kernel(a, b):
    # We want to take products of non-infection kernels here
    # Cutoff at .9999 risk to suppress infinities
    return torch.log(1 - risk_kernel(a, b).clamp(None, .9999))

def nbody_problem(pop):
    prob = arrdict.arrdict(
        sources=pop,
        targets=pop,
        charges=np.zeros(len(pop))
    ).map(torch.as_tensor).float().cuda()
    prob['kernel'] = kernel
    return prob

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


def render(charges, next, increment, points, threshold=1e-1, res=1000):
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
    means = sp.ndimage.gaussian_filter(means, .025/scale*res)
    ax.imshow(means, extent=(l, r, b, t))
    
    return fig

def run(n=10e3):
    pop = pop_points(n=n)
    prob = nbody_problem(pop)
    presoln = adaptive.presolve(prob)

    # Set patient zero
    presoln.scaled.charges[0] = 1.

    infected = []
    for t in tqdm(range(120)):
        infected.append(presoln.scaled.charges.cpu().numpy())

        log_nonrisk = adaptive.evaluate(**presoln)
        risk = 1 - torch.exp(log_nonrisk)
        
        rands = torch.rand_like(risk)
        presoln.scaled.charges = ((rands < risk) | (0 < presoln.scaled.charges)).float()
        print(presoln.scaled.charges.sum())

    smoothing = 4
    points = prob.targets.cpu().numpy()
    firsts = [i for i in infected[:-1] for _ in range(smoothing)]
    seconds = [i for i in infected[1:] for _ in range(smoothing)]
    increment = np.arange(0, len(infected)-1, 1/4) % 1
    encoder = recording.parallel_encode(render, firsts, seconds, increment, points=points, N=16, fps=4*smoothing)
    recording.notebook(encoder)
