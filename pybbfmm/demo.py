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
    return .001 * 1/(1 + (d/4)**3)

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

def render(charges, points, threshold=1e-2, eps=.1, res=1000):
    lims = np.stack([
        points[charges > threshold].min(0) - eps,
        points[charges > threshold].max(0) + eps])

    visible = (points > lims[0]).all(-1) & (points < lims[1]).all(-1)

    points, charges = points[visible], charges[visible]

    fig, ax = plt.subplots()
    ax.set_aspect(1)
    fig.set_size_inches(16, 16)
    if len(points) < 2000:
        ax.scatter(*points.T, c=charges, s=1)
    else:
        xy = (res*(points - lims[0])/(lims[1] - lims[0]))
        ij = np.stack([res - xy[:, 1], xy[:, 0]], -1).astype(int).clip(0, res-1)

        sums = np.zeros((res, res))
        np.add.at(sums, tuple(ij.T), charges)

        counts = np.zeros((res, res))
        np.add.at(counts, tuple(ij.T), np.ones_like(charges))

        means = np.full((res, res), np.nan)
        np.divide(sums, counts, out=means, where=counts > 0)

        ax.imshow(means, extent=(*lims[:, 0], *lims[:, 1]))
    
    return fig

def run(n=10e3):
    pop = pop_points(n=n)
    prob = nbody_problem(pop)
    presoln = adaptive.presolve(prob)

    # Set patient zero
    presoln.scaled.charges[0] = 1.

    risks = []
    for t in tqdm(range(10)):
        log_nonrisk = adaptive.evaluate(**presoln)
        risk = 1 - torch.exp(log_nonrisk)
        
        rands = torch.rand_like(risk)
        presoln.scaled.charges = (rands < risk).float()
        risks.append(risk.cpu().numpy())

    encoder = recording.parallel_encode(render, risks, points=prob.targets.cpu().numpy(), N=0, fps=1)
    recording.notebook(encoder)