import numpy as np
import matplotlib.pyplot as plt

def cartesian_product(xs, D):
    return np.stack(np.meshgrid(*((xs,)*D), indexing='ij'), -1)

def flat_cartesian_product(xs, D):
    return cartesian_product(xs, D).reshape(-1, D)

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
            self._nodes = flat_cartesian_product(onedim, self.D)
        return self._nodes

    def zeros(self):
        return np.zeros(self.N**self.D)

    def similarity(self, a, b):
        """
        Args:
            a: ms + (d,)
            b: ns + (d,)
        
        Returns:
            ms + ns
        """
        assert ((-1 <= a) & (a <= +1)).all()
        assert ((-1 <= b) & (b <= +1)).all()

        da = a.shape[:-1] + (1,)*(b.ndim-1) + (self.D, 1)
        db = (1,)*(a.ndim-1) + b.shape[:-1] + (self.D, 1)

        theta_a = np.arccos(a).reshape(da)
        theta_b = np.arccos(b).reshape(db)

        ks = np.arange(1, self.N)[None, None, None, :]
        terms = np.cos(ks*theta_a)*np.cos(ks*theta_b)
        return (1/self.N + 2/self.N*terms.sum(-1)).prod(-1)

    def interpolate(self, x, v):
        return (self.similarity(x, self.nodes)*v).sum(-1)

    def anterpolate(self, x, v):
        return (self.similarity(self.nodes, x)*v).sum(-1)

def test_similarity():

    def g(x):
        return (x**5).sum(-1)

    ns = nodes(N, 2)
    xs = np.linspace(-1, +1, 101)[:, None]
    ghat = (similarity(xs, ns)*g(ns)).sum(-1)

    plt.plot(xs[:, 0], g(xs))
    plt.plot(xs[:, 0], ghat)

