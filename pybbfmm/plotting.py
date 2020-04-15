import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from . import orthantree

def plot_tree(tree, ax=None, color={}, number=False):
    tree = tree.cpu().numpy()

    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.set_xlim(-1.1, +1.1)
    ax.set_ylim(-1.1, +1.1)
    ax.set_aspect(1)

    color = {int(v): k for k, vs in color.items() for v in vs}

    for id, center, depth in zip(tree.id, tree.centers, tree.depths):
        if id in color:
            kwargs = {'color': color[id], 'fill': True, 'alpha': .25} 
        else:
            kwargs = {'color': 'k', 'fill': False, 'alpha': .25}
        if number:
            ax.annotate(str(id), center, ha='center', va='center')
        width = 2/2**depth
        corner = center - np.array([1, 1])*width/2
        ax.add_artist(mpl.patches.Rectangle(corner, width, width, **kwargs))
            
    return ax

def plot_problem(prob, q=.01, ax=None):
    prob = prob.cpu().numpy()
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.set_aspect(1)

    ax.scatter(*prob.targets.T, color='C0', label='targets', marker='.')

    charges = (prob.charges - prob.charges.min())/(prob.charges.max() - prob.charges.min() + 1e-6)
    ax.scatter(*prob.sources.T, color='red', s=10 + 100*charges, label='sources', marker='x')

    return ax

def plot_lists(tree, lists, b=None):
    b = int(tree.terminal.nonzero().squeeze(-1)[0]) if b is None else b
    color = {'u': 'C0', 'v': 'C1', 'w': 'C2', 'x': 'C3', 'y': 'C4'}
    key = {k: mpl.patches.Rectangle((0., 0.), 1, 1, color=c, alpha=.25) for k, c in color.items()}
    color = {color[k]: l[l[:, 0] == b, 1] for k, l in lists.items()}
    color['C4'] = orthantree.y_list(tree, b)
    ax = plot_tree(tree, color=color)
    ax.legend(key.values(), key.keys())

