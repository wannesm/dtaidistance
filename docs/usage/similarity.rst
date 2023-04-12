Similarity vs Distance
----------------------

Distances such as Euclidean distance or Dynamic Time Warping (DTW)
return a value that expresses *how far two instances are apart*.
Such a distance is equal to zero, when the instances are equal, or larger than
zero. In certain cases you might need to translate this distance to:

- A *similarity measure* that inverts the meaning of the returned
  values and expresses *how close to instances are*. Typically also
  bounded between 0 and 1, where now 1 means that two instances are equal.

- A *bounded distance* that limits the range of the distance between
  0 and 1, where 0 means that two instances are equal. This can be achieved
  by squashing to distance between 0 and 1.

The DTAIDistance toolbox provides a number of transformations to
translate a distance to a similarity measure or to a squashed distance.

Similarity
~~~~~~~~~~

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

When reapplying the distance_to_similarity function over multiple matrices, it is advised
to set the r argument manually (or extract them using the return_params
option). Otherwise they are computed based on
the given distance matrix and will be different from call to call.

Squashing
~~~~~~~~~

Similarity reverses high values to low and low to high. If you want to
maintain the direction but squash the distances between 0 and 1, you can
use the squash function (based on Vercruyssen et al., Semi-supervised anomaly detection with an application to
water analytics, ICDM, 2018).

.. code-block:: python

    similarity.squash(dtw.distance_matrix(s))

Which results in:

.. code-block:: python

    [[0.00 0.75 0.99 0.00 0.75 0.99]
     [0.75 0.00 0.94 0.75 0.00 0.94]
     [0.99 0.94 0.00 0.99 0.94 0.00]
     [0.00 0.75 0.99 0.00 0.75 0.99]
     [0.75 0.00 0.94 0.75 0.00 0.94]
     [0.99 0.94 0.00 0.99 0.94 0.00]]

You can observe the diagonal is all zeros again (when rounded, the values
are slightly larger than zero because logistic squashing is used). And
the most different series are close to 1.

When reapplying the squash function over multiple matrices, it is advised
to set the x0 and r argument manually (or extract them using the return_params
option). Otherwise they are computed based on
the given distance matrix and will be different from call to call.
