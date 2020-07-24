########
Concepts
########

There are some concepts that show up in many places in the pybbfmm docs, so they're collected together here.

.. _dotdicts:

arrdicts and dotdicts
*********************
dotdicts and arrdicts are somewhere between dictionaries and classes, and are a crutch for research code that I'm
immensely fond of. The :ref:`megastep docs have a lot more detail <megastep:dotdicts>`.

.. _problem:

Problems
********
A problem is a :ref:`dotdict <dotdicts>` that describes the problem you're trying to solve. A problem dotdict has 
keys 

sources
    A (n_sources, D)-tensor giving the D-dimensional locations of the source charges.
targets
    A (n_targets, D)-tensor giving the D-dimensional locations of the target points you want to evaluate the field at.
charges
    A (n_sources,)-tensor giving the 'charge' at each source point. 
kernel
    A function that takes a tensor of source locations and a tensor of target locations and returns the strength of the 
    sources' fields at the target points. For example, an inverse-square law uses a kernel ::

        def quad_kernel(a, b):
            return 1/((a - b)**2).sum(-1)

One important restriction is that **multiple targets and sources should not have the exact same location**. Otherwise
the tree construction might not terminate. 

.. _presolve:

Presolve: trees, indices & schemes
**********************************
The :func:`~pybbfmm.presolve` computes a bunch of metadata based on only the positions of the problem's points and the 
kernel used. This presolve metadata can then be used for as many :ref:`evaluations <eval>` on different charges as you
like.

If you ever get confused about any presolve metadata, the best way to figure things out is to generate the metadata
for a low-capacity, 1D problem::

    from pybbfmm import *
    from pybbfmm.test import *

    prob = arrdict.arrdict(
        sources=np.array([[-1], [+.5], [+1]]),
        charges=np.zeros((3,)),
        targets=np.array([[0.]])
    ).map(lambda t: torch.as_tensor(t).float())
    prob['kernel'] = quad_kernel

    presolve(prob)

Trees
-----
The most important bit of presolve information is the *tree*. The tree is the binary partition of space that's used 
to figure out which sources it's okay to approximate when calculating which points.

The internal nodes of the tree are called *boxes*. The boxes are represented as an index, with the root being index 0. 
This means that you'll usually find the attribute of box ``i`` at index ``i`` of a tensor: ``parents[3]`` gives the
index of the parent of ``3``, ``children[3]`` gives the children of ``3``, etc etc.

The tree is represented as an :ref:`arrdict <dotdicts>` of tensors. Each tensor is n_boxes long, and gives a different
property of the boxes:

``id``
    Give the boxes' ID, which is just... its index. Useful to have around, not actually informative.

``parents``
    Gives the boxes' parents. The ``i`` th element is the index of box ``i``s parent.

``depths``
    Gives the boxes' depth in the tree. The ``i`` th element is the depth of box ``i``, with zero corresponding to the root.

``centers``
    Gives coordinates of the boxes' centers. The ``i`` th element... you get the idea.

``terminal``
    Gives a boolean saying whether that box is a leaf. True means the box is a leaf; that it has no children.

``children``
    Gives the boxes' children. The ``i``th element is an (2,)/(2, 2)/(2, 2, 2)/etc-tensor of the 2 :sup:`D` children of box ``i``.

``descent``
    Gives the boxes' descent: whether it's a left-child or a right-child, etc etc. Practically this means the ``i``th
    element is a (D,)-vector of (-1, +1)'s, with the value indicating which side of the parent's center the box is on.

Indices
-------
An arrdict mapping sources and targets to the leaf box they lie in.

Depths
------
A ragged array mapping each depth to the boxes at that depth.

Schemes
-------
    
 
.. _eval:

Evaluate: weights, interactions & contributions
***********************************************

Weights
-------

Interactions
------------

Contributions
-------------

In brief, the u-list of a leaf is the set of neighbouring leaves. 

In brief, the v-list of a box is the children of the parent's colleagues that are separated from the box

In brief, the w-list of a leaf is the set of descendents of colleagues whose parents are adjacent but which aren't
themselves

See Carrier, Greengard & Rokhlin's 1988 paper for a description of u, v, w, and x interactions:

https://pdfs.semanticscholar.org/97f0/d2a31d818ede922c9a59dc17f710642332ca.pdf

§3.2, Notation, is what you're after, along with Fig 5.