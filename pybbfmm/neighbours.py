import numpy as np
from types import SimpleNamespace

def neighbour(tree, node, direction):
    """Finds the neighbour of a node in an adaptive quadtree or it's D-dimensional
    generalizations.
    
    Args:
        tree: an object holding a bunch of node attributes, indexed by node:
            * `parent`, an (N,)-array of integers. The `n`th element gives the 
               index of `n`'s parent. The parent of the root is -1.
            * `children`, an ((N,) + (2,)*D)-array of integers. The `n`th slice 
               gives the indices of `n`'s children. The bottom-left child of node 3
               in a quadtree would be (3, 0, 1). 
            * `descent`, an (N, D)-array with elements from {-1, +1}. The `n`th
               row gives which direction node `n` lies in compared to its parent.
               For example, the top-left quadrant in a quadtree would be `(-1, +1)`.
            * `terminal`, an (N,)-array of booleans. The `n`th element is True
               if node `n` is a leaf.
        node: an integer, the index of the node you want to find the neighbour of.
        direction: a (D,)-array with elements from {-1, +1}
    
    Returns:
        An integer giving the index of the neighbouring node, or -1 if it doesn't 
        exist.
    """
    direction = np.asarray(direction)

    # Ascend to the common ancestor
    neighbour_descents = []
    while True:
        node_descent = tree.descent[node]
        neighbour_descent = node_descent*(1 - 2*abs(direction))
        neighbour_descents.append(neighbour_descent)

        direction = ((node_descent + direction)/2).astype(int)
        node = tree.parent[node]
        if (direction == 0).all() or node < 0:
            break
            
    # Descend to the neighbour 
    for neighbour_descent in neighbour_descents[::-1]:
        if (tree.terminal[node]) or (node < 0):
            break
        node = tree.children[(node, *(neighbour_descent.T + 1)//2)]
  
    return node

def random_tree(term_prob=.5, D=2, min_size=10, max_size=100):
    if min_size:
        while True:
            tree = random_tree(term_prob, D, None, max_size)
            if len(tree.parent) > min_size:
                return tree

    tree = SimpleNamespace(
        parent=np.array([-1]),
        children=np.full((1,)+(2,)*D, -1),
        descent=np.full((1, D), 0),
        terminal=np.array([False]),)
    active = np.array([0])

    while True:
        prob = term_prob if len(tree.parent) < max_size else 1.
        terminated = np.random.rand(len(active)) < prob
        tree.terminal[active[terminated]] = True

        active = active[~terminated]
        if active.size == 0:
            break

        tree.parent = np.r_[tree.parent, active.repeat(2**D)]

        children = len(tree.children) + np.arange(2**D * len(active))
        tree.children[active] = children.reshape(len(active), *(2,)*D)
        tree.children = np.r_[tree.children, np.full((len(children), *(2,)*D), -1)]

        descent = np.stack(np.meshgrid(*([-1, +1],)*D, indexing='ij'), -1)
        descent = descent[None].repeat(len(active), 0)
        tree.descent = np.r_[tree.descent, descent.reshape(-1, D)]

        tree.terminal = np.r_[tree.terminal, np.full(len(children), False)]
        active = children
    
    return tree

def plot(tree, color={}, ax=None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    depths, centers = {0: 0}, {0: np.array([0, 0])}
    for node in range(1, len(tree.parent)):
        depths[node] = depths[tree.parent[node]] + 1
        centers[node] = centers[tree.parent[node]] + tree.descent[node]/2.**depths[node]

    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.set_xlim(-1.1, +1.1)
    ax.set_ylim(-1.1, +1.1)
    ax.set_aspect(1)

    for node in depths:
        width = 2/2**depths[node]
        corner = centers[node] - np.array([1, 1])*width/2
        if node in color:
            kwargs = {'color': color[node], 'fill': True, 'alpha': .5}
        else: 
            kwargs = {'color': 'k', 'fill': False}

        ax.add_artist(mpl.patches.Rectangle(corner, width, width, **kwargs))

def test(tree, repeats=10):
    depths, centers = {0: 0}, {0: np.array([0, 0])}
    for node in range(1, len(tree.parent)):
        depths[node] = depths[tree.parent[node]] + 1
        centers[node] = centers[tree.parent[node]] + tree.descent[node]/2.**depths[node]

    for _ in range(repeats):
        direction = np.random.choice([-1, 0, +1], size=(2,))
        A = np.random.randint(len(tree.parent))
        B = neighbour(tree, A, direction)

        A_boundary = centers[A] + direction/2**depths[A]
        if abs(A_boundary).max() >= 1:
            assert B == -1
        else:
            B_boundary = centers[B] - direction/2**depths[B]
            assert (A_boundary - B_boundary) @ direction < 1e-6

def run():
    pass
