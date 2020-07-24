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
A problem is a :ref:`dotdict <dotdicts>` that describes the problem you're trying to solve.

.. _presolve:

Presolve: trees, indices & schemes
**********************************


Tree
----
An arrdict describing the tree itself. Thinking of indexing into the arrays in this dict as a map of sorts,
    * ``parents``: maps boxes to their parents
    * ``depths``: maps boxes to their depth in the tree
    * ``centers``: maps boxes to their physical center
    * ``terminal``: maps boxes to a boolean saying whether that box is a leaf
    * ``children``: maps boxes to a (2,)/(2, 2)/(2, 2, 2)/etc array of 2**D children
    * ``descent``: maps boxes to a (D,)-vector of what kind of child that box is, with elements from (-1, +1).

The boxes are represented as an index, with the root being index 0. This means that you'll usually find the attribute
of box ``i`` at index ``i`` of an array: ``parents[3]`` gives the index of the parent of ``3``, ``children[3]`` gives the
children of ``3``, etc etc.

Indices
-------
An arrdict mapping sources and targets to the leaf box they lie in.

Depths
------
A ragged array mapping each depth to the boxes at that depth.
    
 
.. _eval:

Evaluate: weights, interactions & contributions
***********************************************
In brief, the u-list of a leaf is the set of neighbouring leaves. 

In brief, the v-list of a box is the children of the parent's colleagues that are separated from the box

In brief, the w-list of a leaf is the set of descendents of colleagues whose parents are adjacent but which aren't
themselves

See Carrier, Greengard & Rokhlin's 1988 paper for a description of u, v, w, and x interactions:

https://pdfs.semanticscholar.org/97f0/d2a31d818ede922c9a59dc17f710642332ca.pdf

§3.2, Notation, is what you're after, along with Fig 5.