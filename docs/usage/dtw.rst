Dynamic Time Warping (DTW)
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis
    import numpy as np
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
    s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
    path = dtw.warping_path(s1, s2)
    dtwvis.plot_warping(s1, s2, path, filename="warp.png")

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/dtw_example.png?v=5
   :alt: DTW Example


DTW Distance Measure Between Two Time Series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Only the distance measure based on two sequences of numbers:

::

    from dtaidistance import dtw
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    distance = dtw.distance(s1, s2)
    print(distance)

The fastest version (30-300 times) uses c directly but requires an array
as input (with the double type), , and (optionally) also prunes computations
by setting ``max_dist`` to the Euclidean upper bound:

::

    from dtaidistance import dtw
    import array
    s1 = array.array('d',[0, 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = array.array('d',[0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2, use_pruning=True)

Or you can use a numpy array (with dtype double or float):

::

    from dtaidistance import dtw
    import numpy as np
    s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double)
    s2 = np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0], dtype=np.double)
    d = dtw.distance_fast(s1, s2, use_pruning=True)

Check the ``__doc__`` for information about the available arguments:

::

    print(dtw.distance.__doc__)


DTW Complexity and Early-Stopping
"""""""""""""""""""""""""""""""""

The ``distance`` function has linear space complexity but quadratic
time complexity. To reduce the time complexity a number of options
are available. The most used approach accros DTW implementations is
to use a window that indicates the maximal shift that is allowed.
This reduces the complexity to the product of window size and
largest sequence length:

-  ``window``: Only allow for shifts up to this amount away from the two
   diagonals.

A number of other options are foreseen to early stop some or all paths the
dynamic programming algorithm is exploring:

-  ``max_dist``: Avoid computing partial paths that will be larger
   than this value. If no solution is found that is smaller or equal
   to this value, then return infinity.
-  ``use_pruning``: A good way of pruning partial paths is to set ``max_dist`` to the
   Euclidean upper bound. If this option is true, this is done automatically.
-  ``max_step``: Do not allow steps larger than this value, replace them
   with infinity.
-  ``max_length_diff``: Return infinity if difference in length of two
   sequences is larger.


DTW Tuning
""""""""""

A number of options are foreseen to tune how the cost is computed:

-  ``penalty``: Penalty to add if compression or expansion is applied
   (on top of the distance).
-  ``psi``: Up to ``psi`` number of start and end points of a sequence can be
   ignored if this would lead to a lower distance. This is also called
   psi-relaxation (for cyclical sequences) [2].


DTW and keep all warping paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If, next to the distance, you also want the full matrix to see all
possible warping paths:

::

    from dtaidistance import dtw
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    distance, paths = dtw.warping_paths(s1, s2)
    print(distance)
    print(paths)

The matrix with all warping paths can be visualised as follows:

::

    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis
    import random
    import numpy as np
    x = np.arange(0, 20, .5)
    s1 = np.sin(x)
    s2 = np.sin(x - 1)
    random.seed(1)
    for idx in range(len(s2)):
        if random.random() < 0.05:
            s2[idx] += (random.random() - 0.5) / 2
    d, paths = dtw.warping_paths(s1, s2, window=25, psi=2)
    best_path = dtw.best_path(paths)
    dtwvis.plot_warpingpaths(s1, s2, paths, best_path)

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/warping_paths.png?v=3
   :alt: DTW Example


Notice the ``psi`` parameter that relaxes the matching at the beginning
and end. In this example this results in a perfect match even though the
sine waves are slightly shifted.

DTW between multiple Time series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compute the DTW distance measures between all sequences in a list of
sequences, use the method ``dtw.distance_matrix``. You can speed up the
computation by using the ``dtw.distance_matrix_fast`` method that tries
to run all algorithms in C. Also parallelization can be activated using
the ``parallel`` argument.

The ``distance_matrix`` and ``distance_matrix_fast`` methods expect a
list of lists/arrays:

::

    from dtaidistance import dtw
    import numpy as np
    timeseries = [
        np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double),
        np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0.0, 0, 1, 2, 1, 0, 0, 0])]
    ds = dtw.distance_matrix_fast(timeseries)

or a matrix (in case all time series have the same length):

::

    from dtaidistance import dtw
    import numpy as np
    timeseries = np.array([
        [0.0, 0, 1, 2, 1, 0, 1, 0, 0],
        [0.0, 1, 2, 0, 0, 0, 0, 0, 0],
        [0.0, 0, 1, 2, 1, 0, 0, 0, 0]])
    ds = dtw.distance_matrix_fast(timeseries)

The result is stored in a matrix representation. Since only the upper
triangular matrix is required this representation more memory then necessary.
This behaviour can be deactivated by setting the argument ``compact`` to
true. The method will then return a 1-dimensional array with all results.
This array represents the concatenation of all upper triangular rows.


DTW between multiple time series, limited to block
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can instruct the computation to only fill part of the distance
measures matrix. For example to distribute the computations over
multiple nodes, or to only compare source time series to target time series.

::

    from dtaidistance import dtw
    import numpy as np
    timeseries = np.array([
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]])
    ds = dtw.distance_matrix_fast(timeseries, block=((1, 4), (3, 5)))

The output in this case will be:

::

    #  0     1    2    3       4       5
    [[ inf   inf  inf     inf     inf  inf]    # 0
     [ inf   inf  inf  1.4142  0.0000  inf]    # 1
     [ inf   inf  inf  2.2360  1.7320  inf]    # 2
     [ inf   inf  inf     inf  1.4142  inf]    # 3
     [ inf   inf  inf     inf     inf  inf]    # 4
     [ inf   inf  inf     inf     inf  inf]]   # 5

Especially for blocks the matrix representation uses a lot of unnecesary
memory. This can be avoided by setting the ``compact`` argument to true:

::

    from dtaidistance import dtw
    import numpy as np
    timeseries = np.array([
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]])
    ds = dtw.distance_matrix_fast(timeseries, block=((1, 4), (3, 5)), compact=True)

The result will now be:

::

    [1.4142  0.0000  2.2360  1.7320  1.4142]


DTW based on shape (z-normalization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are interested in comparing only the shape, and not the absolute
differences and offset, you need to z-normalize the data first. This can be achieved
using the SciPy ``zscore`` function:

::

    import numpy as np
    a = np.array([0.1, 0.3, 0.2, 0.1])
    from scipy import stats
    az = stats.zscore(a)
    # az = array([-0.90453403,  1.50755672,  0.30151134, -0.90453403])

Multi-dimensionsal DTW
^^^^^^^^^^^^^^^^^^^^^^

Compare two multi-dimensional sequences.

Assumes the first dimension to be the sequence item index, and the second
dimension to be the series index (thus timestep).

Example:

::

    s1 = np.array([[0, 0],
                   [0, 1],
                   [2, 1],
                   [0, 1],
                   [0, 0]], dtype=np.double)
    s2 = np.array([[0, 0],
                   [2, 1],
                   [0, 1],
                   [0, .5],
                   [0, 0]], dtype=np.double)
    d = distance(s1, s2)

This method returns the dependent DTW (DTW_D) distance between two
n-dimensional sequences. If you want to compute the independent DTW
(DTW_I) distance, use the 1-dimensional version:

::

    dtw_i = 0
    for dim in range(ndim):
        dtw_i += dtw.distance(s1[:,dim], dtw.distance(s2[:,dim])
