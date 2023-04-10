Similarity
----------

Instead of expressing a distance, thus how far two instances are apart,
one can also express a similarity, how close two instances are.
Whereas a distance is larger than zero and have no upperbound,
similarity is between 0 and 1.

Some methods require as input a similarity instead of a distance
(e.g., spectral clustering). Therefore, it might be useful to translate
the computed distances to a similarity. There are different approaches
to achieve this that are supported by dtaidistance: exponential,
Gaussian, reciprocal, reverse.

For example, given a set of series (the rows) for which we want to compute the
pairwise similarity based on dynamic time warping:

.. code-block:: python

    from dtaidistance import dtw, similarity
    s = np.array([[0., 0, 1, 2, 1, 0, 1, 0, 0],
                  [0., 1, 2, 0, 0, 0, 0, 0, 0],
                  [1., 2, 0, 0, 0, 0, 0, 1, 1],
                  [0., 0, 1, 2, 1, 0, 1, 0, 0],
                  [0., 1, 2, 0, 0, 0, 0, 0, 0],
                  [1., 2, 0, 0, 0, 0, 0, 1, 1]])
    sim = similarity.distance_to_similarity(dtw.distance_matrix(s))

The result is:

.. code-block:: python

    [[1.00 0.53 0.37 1.00 0.53 0.37]
     [0.53 1.00 0.46 0.53 1.00 0.46]
     [0.37 0.46 1.00 0.37 0.46 1.00]
     [1.00 0.53 0.37 1.00 0.53 0.37]
     [0.53 1.00 0.46 0.53 1.00 0.46]
     [0.37 0.46 1.00 0.37 0.46 1.00]]

You can observe that the diagonal is all ones because each series
is similar to itself. And the series at index 0 and 3 are identical,
thus also resulting in a similarity of 1.

If you want to use a different conversion than the default exponential
by using the method argument.

.. code-block:: python

    distance_to_similarity(distances, method='exponential')
    distance_to_similarity(distances, method='gaussian')
    distance_to_similarity(distances, method='reciprocal')
    distance_to_similarity(distances, method='reverse')

