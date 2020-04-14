import numpy as np
from types import SimpleNamespace

def neighbours(tree, node, direction):
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
    # Ascend to the common ancestor
    neighbour_descents = []
    while True:
        node_descent = tree.descent[node]
        neighbour_descent = node_descent*(1 - 2*direction.abs())
        neighbour_descents.append(neighbour_descent)

        direction = int((node_descent + direction)/2)
        node = tree.parent[node]
        if (direction == 0).all() or node < 0:
            break
            
    # Descend to the neighbour 
    for neighbour_descent in neighbour_descents[::-1]:
        if (tree.terminal[node]) or (node == -1):
            break
        node = tree.children[(node, *(neighbour_descent.T + 1)/2)]
  
    return node

def random_tree(term_prob=.75, D=2):
    tree = SimpleNamespace(
        parent=np.array([-1]),
        children=np.full((1,)+(2,)*D, -1),
        descent=np.full((1, D), -1),
        terminal=np.array([False]),)
    active = np.array([0])

    while True:
        terminated = np.random.rand(len(active)) < term_prob
        tree.terminal[active[terminated]] = True

        active = active[~terminated]
        if active.size == 0:
            break

        tree.parent = np.r_[tree.parent, active.repeat(2**D)]

        children = len(tree.children) + np.arange(2**D * len(active))
        tree.children = np.r_[tree.children, children.reshape(len(active), *(2,)*D)]

        descent = np.stack(np.meshgrid(*([-1, +1],)*D), -1)
        descent = descent[None].repeat(len(active)).reshape(-1, D)
        tree.descent = np.r_[tree.descent, descent]

        tree.terminal = np.r_[tree.terminal, np.full(len(children), False)]
        active = children
    
    return tree




