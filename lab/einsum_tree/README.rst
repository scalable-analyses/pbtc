Einsum Trees
============

This section expands the capabilities of our tensor compiler by adding support for einsum trees.
Specifically, we execute einsum trees by mapping them to a tree of unary and binary tensor operations.
These operations can then be executed by our tensor operation backend.

Lowering
--------

An einsum tree can represent multiple dependent tensor operations.
It may contain nodes with two children (contractions), nodes with one child (transpositions), or leaf nodes (input tensors).
The output tensor of the root node represents the result of the entire tree.

First Example
^^^^^^^^^^^^^

Einsum tree:

.. code-block::

   0,1,2,3,4
   ├─ 7,3,4
   |  ├─ 8,4
   |  └─ 7,3,8
   └─ 0,1,2,7
      ├─ 1,2,5,7
      |  ├─ 2,6,7
      |  └─ 1,5,6
      └─ 0,5

String representation:

.. code-block::

   [[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]

Dimension sizes (sorted by numerical ID):

.. code-block::

   100,72,128,128,3,71,305,32,3

Second Example
^^^^^^^^^^^^^^

Einsum tree:

.. code-block::

   0,1,2,3
   ├─ 0,4,7,8,2,3
   |  ├─ 7,8,5,6,2,3
   |  |  ├─ 8,6,9,3
   |  |  |  └─ 3,6,8,9
   |  |  └─ 7,5,2,9
   |  |     └─ 2,5,7,9
   |  └─ 0,4,5,6
   └─ 1,4,7,8

String representation:

.. code-block::

   [[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]

Dimension sizes (sorted by numerical ID):

.. code-block::

   60,60,20,20,8,8,8,8,8,8

.. admonition:: Tasks

    1. Implement a function that parses the string representation of a tree and the numerically sorted dimension sizes.
    2. Implement a function that lowers the contraction and permutation nodes of the tree to your tensor operation backend.
    3. Run your optimization passes on the lowered tensor operations.
    4. Benchmark the performance of your implementation for the above examples. Report the measured performance in GFLOPS.

Optimization
------------

This section optimizes einsum trees using three transformations:

1. Reorder the index string of an interior node.
2. Swap the operands of a contraction node.
3. Insert a permutation node.

First Example
^^^^^^^^^^^^^

Einsum tree:

.. code-block::

   0,1,2,3,4
   ├─ 7,3,4
   |  ├─ 7,3,8
   |  └─ 8,4
   └─ 0,1,2,7
      ├─ 0,5
      └─ 5,1,2,7
         ├─ 5,1,6
         └─ 6,2,7

String representation:

.. code-block::

   [[7,3,8],[8,4]->[7,3,4]],[[0,5],[[5,1,6],[6,2,7]->[5,1,2,7]]->[0,1,2,7]]->[0,1,2,3,4]

Dimension sizes (by numerical ID):

.. code-block::

   100,72,128,128,3,71,305,32,3

Second Example
^^^^^^^^^^^^^^

Einsum tree:

.. code-block::

   0,1,2,3
   ├─ 1,4,7,8
   └─ 0,4,2,7,3,8
      ├─ 0,4,5,6
      └─ 2,5,7,3,6,8
         ├─ 2,5,7,9
         └─ 3,6,8,9

String representation:

.. code-block::

  [1,4,7,8],[[0,4,5,6],[[2,5,7,9],[3,6,8,9]->[2,5,7,3,6,8]]->[0,4,2,7,3,8]]->[0,1,2,3]

Dimension sizes (by numerical ID):

.. code-block::

   60,60,20,20,8,8,8,8,8,8

Third Example
^^^^^^^^^^^^^

.. code-block::

   5,6,7,8,9
   ├─ 2,7,8,4
   |  ├─ 2,7,3
   |  └─ 3,8,4
   └─ 4,9,5,6,2
      ├─ 4,9,0
      └─ 0,5,6,2
         ├─ 0,5,1
         └─ 1,6,2

String representation:

.. code-block::

  [[2,7,3],[3,8,4]->[2,7,8,4]],[[4,9,0],[[0,5,1],[1,6,2]->[0,5,6,2]]->[4,9,5,6,2]]->[5,6,7,8,9]

Dimension sizes (by numerical ID):

.. code-block::

   40,40,40,40,40,25,25,25,25,25

.. admonition:: Tasks

   1. Develop an optimization pass for einsum trees that applies the three transformations.
   2. Benchmark the performance of your implementation on the provided examples. Report the measured performance in GFLOPS.