import aljpy
import numpy as np
import matplotlib.pyplot as plt
from . import test, chebyshev

class Grids:

    def __init__(self, depth, dim, tail=(), dtype=float):
        n = ((2**dim)**depth - 1)/((2**dim) - 1)
        self._storage = np.zeros((n, *tail), dtype=dtype)
