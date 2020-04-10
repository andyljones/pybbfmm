import torch
import numpy as np
import matplotlib.pyplot as plt

def cartesian_product(xs, D):
    xs = torch.as_tensor(xs)
    return torch.stack(torch.meshgrid(*((xs,)*D)), -1)

def flat_cartesian_product(xs, D):
    return cartesian_product(xs, D).reshape(-1, D)

class Chebyshev:
    
    def __init__(self, N, D, device=None):
        self.N = N
        self.D = D
        self._nodes = None
        self.device = device

    @property
    def nodes(self):
        if self._nodes is None:
            ms = torch.arange(self.N, device=self.device)
            onedim = torch.cos((ms+1/2)*np.pi/self.N)
            self._nodes = flat_cartesian_product(onedim, self.D)
        return self._nodes

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

        theta_a = torch.acos(a).reshape(da)
        theta_b = torch.acos(b).reshape(db)

        ks = torch.arange(1, self.N, device=a.device)[None, None, None, :]
        terms = torch.cos(ks*theta_a)*torch.cos(ks*theta_b)
        return (1/self.N + 2/self.N*terms.sum(-1)).prod(-1)

    def interpolate(self, x, v):
        return (self.similarity(x, self.nodes)*v).sum(-1)

    def anterpolate(self, x, v):
        return (self.similarity(self.nodes, x)*v).sum(-1)

def test_similarity():

    cheb = Chebyshev(4, 1)

    def g(x):
        return (x**5).sum(-1)

    ns = cheb.nodes
    xs = torch.linspace(-1, +1, 101)[:, None]
    ghat = (cheb.similarity(xs, ns)*g(ns)).sum(-1)

    plt.plot(xs[:, 0], g(xs))
    plt.plot(xs[:, 0], ghat)

