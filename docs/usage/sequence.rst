Sequences
---------

When the values in the time series are symbols rather than numbers dynamic programming
can be used to find the globally optimal sequence alignment. This is the same basic
algorithm as dynamic time warping but with a different cost function.
In this toolbox the Needleman-Wunsch algorithm is available.

Needleman-Wunsch sequence alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    s1 = "GATTACA"
    s2 = "GCATGCU"
    from dtaidistance import alignment
    value, matrix = alignment.needleman_wunsch(s1, s2)
    algn, s1a, s2a = alignment.best_alignment(matrix, s1, s2, gap='-')

This will result in the following alignment:

.. code-block:: python

    s1a = 'G-ATTACA'
    s2a = 'GCAT-GCU'

The matrix representing all possible optimal alignments is

.. code-block:: python

    matrix = [
        [ 0, -1, -2, -3, -4, -5, -6, -7],
        [-1,  1,  0, -1, -2, -3, -4, -5],
        [-2,  0,  0,  1, -0, -1, -2, -3],
        [-3, -1, -1,  0,  2,  1,  0, -1],
        [-4, -2, -2, -1,  1,  1,  0, -1],
        [-5, -3, -3, -1,  0,  0,  0, -1],
        [-6, -4, -2, -2, -1, -1,  1,  0],
        [-7, -5, -3, -1, -2, -2,  0,  0]]

