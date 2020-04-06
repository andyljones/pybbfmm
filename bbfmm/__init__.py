import scipy.signal
import aljpy
import numpy as np
import matplotlib.pyplot as plt
from . import test, chebyshev, tree, fftconvolve
from itertools import product

KERNEL = test.quad_kernel
EPS = 1e-2

def limits(prob):
    points = np.concatenate([prob.sources, prob.targets])
    return np.stack([points.min(0) - EPS, points.max(0) + EPS])

def scale(prob):
    lims = limits(prob)
    return aljpy.dotdict(
        limits=lims,
        sources=(prob.sources - lims[0])/(lims[1] - lims[0]),
        charges=prob.charges,
        targets=(prob.targets - lims[0])/(lims[1] - lims[0]))

def leaf_sum(l, x, d):
    x = np.asarray(x)
    D = l.shape[1]
    totals = np.zeros((2**d,)*D + x.shape[1:])
    np.add.at(totals, tuple(l.T), x)
    return totals

def leaf_centers(d):
    return (1/2 + np.arange(2**d))/2**d

def tree_leaves(scaled, cutoff=5):
    D = scaled.sources.shape[1]
    sl = np.zeros((len(scaled.sources), D), dtype=int)
    tl = np.zeros((len(scaled.targets), D), dtype=int)

    #TODO: You can probably get very smart about this and just convert the xs
    # to ints and look at their binary representation.
    d = 0
    while True:
        s_done = leaf_sum(sl, 1, d).max() <= cutoff
        t_done = leaf_sum(tl, 1, d).max() <= cutoff
        if s_done and t_done:
            break

        centers = leaf_centers(d)
        sl = 2*sl + (scaled.sources >= centers[sl]).astype(int)
        tl = 2*tl + (scaled.targets >= centers[tl]).astype(int)

        d += 1

    return aljpy.dotdict(
        sources=sl, 
        targets=tl, 
        depth=d)

def uplift_coeffs(cheb):
    shifts = chebyshev.cartesian_product([-.5, +.5], cheb.D)
    children = shifts[..., None, :] + cheb.nodes/2
    S = cheb.similarity(cheb.nodes, children)
    return np.moveaxis(S, 0, -2)

def weights(scaled, cheb, leaves):
    loc = scaled.sources * 2**leaves.depth - leaves.sources
    S = cheb.similarity(cheb.nodes, 2*loc-1)
    Ws = [leaf_sum(leaves.sources, (S*scaled.charges).T, leaves.depth)]

    coeffs = uplift_coeffs(cheb)
    dot_dims = (
        list(range(1, 2*cheb.D, 2)) + [-1],
        list(range(cheb.D)) + [-1])
    for d in reversed(range(leaves.depth)):
        exp_dims = sum([(s//2, 2) for s in Ws[-1].shape[:-1]], ())
        W_exp = Ws[-1].reshape(*exp_dims, -1)
        Ws.append(np.tensordot(W_exp, coeffs, axes=dot_dims))
    return list(reversed(Ws))

def interactions(W, scaled, cheb):
    D, N = cheb.D, cheb.N
    width = W.shape[0]

    # (nephew offset) + (nephew node) + (child offset) + (child node) + (D,) 
    # (3, 2)*D + (N**D,) + (2,)*D + (N**D,) + (D,)

    child_dims = (1, 1)*D + (1,) + (2,)*D + (1,) + (D,)
    child_offsets = chebyshev.cartesian_product([0, 1], D).reshape(child_dims)

    child_node_dims = (1, 1)*D + (1,) + (1,)*D + (N**D,) + (D,)
    child_nodes = (cheb.nodes/2 + 1/2).reshape(child_node_dims)

    nephew_dims = (3, 2)*D + (1,) + (1,)*D + (1,) + (D,)
    nephew_offsets = chebyshev.cartesian_product(np.arange(-2, 4), D).reshape(nephew_dims)

    nephew_node_dims = (1, 1)*D + (N**D,) + (1,)*D + (1,) + (D,)
    nephew_nodes = (cheb.nodes/2 + 1/2).reshape(nephew_node_dims)

    vectors = ((nephew_offsets + nephew_nodes) - (child_offsets + child_nodes))
    vectors = (scaled.limits[1] - scaled.limits[0])/width*vectors

    nephew_kernel = KERNEL(np.zeros_like(vectors), vectors)

    is_neighbour = (abs(nephew_offsets - child_offsets) <= 1).any(-1)
    interaction_kernel = np.where(is_neighbour, 0, nephew_kernel)
    mirrored = interaction_kernel[(slice(None, None, -1),)*D]

    W_dims = (width//2+2, 2)*D + (N**D,) + (1,)*D + (1,)
    Wp = np.pad(W, (((2, 2),)*D + ((0, 0),)))
    Wp = Wp.reshape(W_dims)

    # Need to disable the valid-dims check in scipy.signal.fftconvolve
    ixns = fftconvolve.fftconvolve(
        Wp,
        mirrored,
        mode='valid',
        axes=np.arange(2*D+1)
    )

    squeeze = tuple(2*d+1 for d in range(D)) + (2*D,)
    ixns = ixns.squeeze(squeeze)

    axes = sum([(i, D+i) for i in range(D)], ()) + (2*D,)
    ixns = ixns.transpose(axes)

    ixns = ixns.reshape((width, width, N**D))

    return ixns


def run():
    prob = test.random_problem(S=100, T=100, D=2)

    cheb = chebyshev.Chebyshev(10, prob.sources.shape[1])

    scaled = scale(prob)
    leaves = tree_leaves(scaled, cutoff=5)

    Ws = weights(scaled, cheb, leaves)
    ixns = interactions(Ws[2], scaled, cheb)

    # Validation
    root = tree.build_tree(prob)
    root.set_weights()
    tree.set_interactions(root)
    root.set_far_field()

    np.testing.assert_allclose(root.W, Ws[0][0, 0])