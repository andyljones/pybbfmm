import torch
import numpy as np
from . import sets

class Chebyshev:
    
    def __init__(self, N, D, device=None):
        self.N = N
        self.D = D
        self._nodes = None
        self.device = device

    @property
    def nodes(self):
        """A cached copy of the locations of the Chebyshev nodes, given as a (N :sup:`D`, D)-tensor.
        """
        if self._nodes is None:
            ms = torch.arange(self.N, device=self.device)
            onedim = torch.cos((ms+1/2)*np.pi/self.N)
            self._nodes = sets.flat_cartesian_product(onedim, self.D)
        return self._nodes

    def similarity(self, a, b):
        """Calculates the Chebyshev 'similarity' between each pair of two sets of points. 

        Don't know if it's called similarity anywhere else, but it's the matrix of weights that lets you go from 
        concrete values to interpolation weights and vice versa. 

        :param a: a (\\*ms, d)-tensor of points
        :param b: a (\\*ns, d)-tensor of points
        :return: a (\\*ms, \\*ns)-tensor of similarities between ``a`` and ``b``'s points.
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

    def upwards_coeffs(self):
        shifts = torch.tensor([-.5, +.5], device=self.device)
        shifts = sets.cartesian_product(shifts, self.D)
        children = shifts[..., None, :] + self.nodes/2
        return self.similarity(self.nodes, children)

    def downwards_coeffs(self):
        shifts = torch.tensor([-.5, +.5], device=self.device)
        shifts = sets.cartesian_product(shifts, self.D)
        children = shifts[..., None, :] + self.nodes/2
        return self.similarity(children, self.nodes)

def test_similarity():
    import matplotlib.pyplot as plt

    cheb = Chebyshev(4, 1)

    def g(x):
        return (x**5).sum(-1)

    ns = cheb.nodes
    xs = torch.linspace(-1, +1, 101)[:, None]
    ghat = (cheb.similarity(xs, ns)*g(ns)).sum(-1)

    plt.plot(xs[:, 0], g(xs))
    plt.plot(xs[:, 0], ghat)

