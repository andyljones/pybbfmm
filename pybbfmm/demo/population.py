import requests
import zipfile
import aljpy
import gzip
import yaml
import numpy as np
from pkg_resources import resource_exists, resource_filename
import pickle
from io import BytesIO

LOGIN = 'https://data-package.ceh.ac.uk/sso/login'
DATA = 'https://data-package.ceh.ac.uk/data/0995e94d-6d42-40c1-8ed4-5090d82471e1.zip'

def credentials():
    # You shouldn't hit this since the result has been folded into the repo.
    # But if you do, go create an account on ceh.ac.uk and then add your
    # details to a file `credentials.yml` in your working directory, with
    # this content:
    # ```
    # CEH:
    #    username: yourusername@gmail.com
    #    password: youractualpassword
    # ```
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

    # This is a hack to allow the demo to be run without CEH credentials
    if resource_exists(__package__, 'density_map.gz'):
        return np.loadtxt(resource_filename(__package__, 'density_map.gz'))

    content = _density_map()
    with zipfile.ZipFile(BytesIO(content)) as zf:
        filename = '0995e94d-6d42-40c1-8ed4-5090d82471e1/data/UK_residential_population_2011_1_km.asc'
        asc = zf.read(filename)
    arr = np.loadtxt(BytesIO(asc), skiprows=6)
    arr[arr == -9999] = np.nan
    return arr

def points(n=1e3):
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