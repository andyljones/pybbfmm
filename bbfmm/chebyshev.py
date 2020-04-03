import numpy as np
import matplotlib.pyplot as plt

class Chebyshev:
    
    def __init__(self, N, D):
        self.N = N
        self.D = D
        self._nodes = None

    @property
    def nodes(self):
        if self._nodes is None:
            ms = np.arange(self.N)
            onedim = np.cos((ms+1/2)*np.pi/self.N)
            multidim = np.stack(np.meshgrid(*([onedim]*self.D), indexing='ij'), -1)
            self._nodes = multidim.reshape(-1, self.D)
        return self._nodes

    def zeros(self):
        return np.zeros(self.N**self.D)

    def _similarity(self, a, b):
        """
        Args:
            a: (m, d)
            b: (j, d)
        
        Returns:
            (m, j)
        """
        assert ((-1 <= a) & (a <= +1)).all()
        assert ((-1 <= b) & (b <= +1)).all()

        theta_a = np.arccos(a)[:, None, :, None]
        theta_b = np.arccos(b)[None, :, :, None]

        ks = np.arange(1, self.N)[None, None, None, :]
        terms = np.cos(ks*theta_a)*np.cos(ks*theta_b)
        return (1/self.N + 2/self.N*terms.sum(-1)).prod(-1)

    def interpolate(self, x, v):
        return (self._similarity(x, self.nodes)*v).sum(-1)

    def anterpolate(self, x, v):
        return (self._similarity(self.nodes, x)*v).sum(-1)

def test_similarity():

    def g(x):
        return (x**5).sum(-1)

    ns = nodes(N, 2)
    xs = np.linspace(-1, +1, 101)[:, None]
    ghat = (similarity(xs, ns)*g(ns)).sum(-1)

    plt.plot(xs[:, 0], g(xs))
    plt.plot(xs[:, 0], ghat)

