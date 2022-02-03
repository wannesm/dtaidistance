Sequences
---------

For time series, it is assumed that it is a sequence of numerical values.
If this is not the case, the same basic algorithm, dynamic programming,
can still be used to find the globally optimal sequence alignment. The
only difference is that it requires a custom cost function.
In this toolbox the Needleman-Wunsch algorithm is available that works
on sequences in general.

Needleman-Wunsch sequence alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    s1 = "GATTACA"
    s2 = "GCATGCU"
    from dtaidistance import alignment
    value, scores, paths = alignment.needleman_wunsch(s1, s2)
    algn, s1a, s2a = alignment.best_alignment(matrix, s1, s2, gap='-')

This will result in the following alignment:

.. code-block:: python

    s1a = 'G-ATTACA'
    s2a = 'GCA-TGCU'

The matrix representing all possible optimal alignments (``paths``) and their
cost (``scores``) is

.. csv-table::
   :header: ,—, G, A,T,T,A,C,A
   :stub-columns: 1

   —,  0 ,    -1 ,     -2 ,    -3 ,    -4 ,     -5 ,   -6 ,    -7
   G, -1 , ↖   1 , ←    0 , ←  -1 , ←  -2 , ←↖  -3 , ← -4 , ←  -5
   C, -2 , ↑   0 , ↖    0 , ↖   1 , ←   0 , ←   -1 , ← -2 , ←  -3
   A, -3 , ↑  -1 , ↑↖  -1 , ↑   0 , ↖   2 , ←    1 , ←  0 , ←  -1
   T, -4 , ↑  -2 , ↑↖  -2 , ↑  -1 , ↑↖  1 , ↖    1 , ←↖ 0 , ←↖ -1
   G, -5 , ↑  -3 , ↑↖  -3 , ↖  -1 , ↑   0 , ↑↖   0 , ↖  0 , ↖  -1
   C, -6 , ↑  -4 , ↖   -2 , ↑  -2 , ↑  -1 , ↑↖  -1, ↖   1 , ←   0
   U, -7 , ↑  -5 , ↑   -3 , ↖  -1 , ←↑ -2 , ↑↖  -2, ↑   0 , ↖   0

If you want to use a custom distance between (some) symbols, you can provide a custom function
using the ``substitution`` argument to  ``needleman_wunsch``. A wrapper is available to translate
a dictionary to a function with:

.. code-block:: python

   substitution_cost = {('A','G'): 2, ('G', 'A'): 3}
   substitution = alignment.make_substitution_fn(substitution_cost)
   value, scores, paths = alignment.needleman_wunsch(s1, s2, substitution=substitution)
