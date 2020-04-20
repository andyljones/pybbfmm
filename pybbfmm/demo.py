import requests
import zipfile
import aljpy
import yaml
import numpy as np
from io import BytesIO

LOGIN = 'https://data-package.ceh.ac.uk/sso/login'
DATA = 'https://data-package.ceh.ac.uk/data/0995e94d-6d42-40c1-8ed4-5090d82471e1.zip'

POPULATION = 65e3

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

def pop_coordinates():
    """Distribute the population around the country, returning a Px2 array of their coordinates relative to the 
    bottom-right corner."""
    density = density_map()
    density = density/np.nansum(density)
    density[np.isnan(density)] = 0.

    # Pick a grid cell for each person in proportion to the density
    valid = (density > 0)
    coords = np.stack(valid.nonzero(), -1)
    ds = density[valid]

    occupancy = np.round(POPULATION*ds).astype(int).clip(1, None)
    indices = np.zeros(occupancy.sum()+1, dtype=int)
    indices[occupancy.cumsum() - occupancy] = 1
    indices = indices.cumsum() - 1

    # Pick a uniformly random spot in the grid cell
    ij = coords[indices, :] + np.random.rand(len(indices), 2)
    xy = np.stack([ij[:, 1], len(density) - ij[:, 0]], -1)

    return xy
