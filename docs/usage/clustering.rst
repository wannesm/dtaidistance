Clustering
----------

Clustering is used to find groups of similar instances (e.g. time series, sequences). Such a
clustering can be used to:

* Identify typical regimes or modes of the source being monitored (see for example
  the `cobras package <https://dtai.cs.kuleuven.be/software/cobras/>`_).
* Identify anomalies, outliers or abnormal behaviour (see for example the
  `anomatools package <https://github.com/Vincent-Vercruyssen/anomatools>`_).

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/hierarchy.png?v=2
   :alt: Clustering hierarchy

Two possible strategies for time series clustering are:

Agglomerative clustering
~~~~~~~~~~~~~~~~~~~~~~~~

A distance matrix can be used for time series clustering. You can use
existing methods such as ``scipy.cluster.hierarchy.linkage`` or one of
two included clustering methods (the latter is a wrapper for the SciPy
linkage method).

::

    # Custom Hierarchical clustering
    model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
    cluster_idx = model1.fit(timeseries)
    # Keep track of full tree by using the HierarchicalTree wrapper class
    model2 = clustering.HierarchicalTree(model1)
    cluster_idx = model2.fit(timeseries)
    # You can also pass keyword arguments identical to instantiate a Hierarchical object
    model2 = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
    cluster_idx = model2.fit(timeseries)
    # SciPy linkage clustering
    model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
    cluster_idx = model3.fit(timeseries)

For models that keep track of the full clustering tree
(``HierarchicalTree`` or ``LinkageTree``), the tree is available in ``model.linkage`` and
can be visualised (see figure at top of this page):

::

    model2.plot("hierarchy.png")

A number of options are also available to tune the layout of the figure. You can also pass your
own set of axes. The only assumption is that the tree is printed to ``ax[0]`` and the
time series to ``ax[1]``.

::

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    show_ts_label = lambda idx: "ts-" + str(idx)
    model.plot("hierarchy.png", axes=ax, show_ts_label=show_ts_label,
               show_tr_label=True, ts_label_margin=-10,
               ts_left_margin=10, ts_sample_length=1)


K-Means DBA clustering
~~~~~~~~~~~~~~~~~~~~~~

K-means clustering for time series requires an averaging strategy for
time series. One possibility is DTW Barycenter Averaging (DBA).

**Example**:

For example, to cluster the `Trace <https://timeseriesclassification.com/description.php?Dataset=Trace>`_
dataset by Davide Roverso.

::

    model = KMeans(k=4, max_it=10, max_dba_it=10, dists_options={"window": 40})
    cluster_idx, performed_it = model.fit(series, use_c=True, use_parallel=False)


.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/kmeans.png?v=2
   :alt: KMeans clustering

**DTW Barycenter Averaging**:

If you only want to run DTW Barycenter Averaging once or multiple times:

::

    new_center = dtw_barycenter.dba(series, center, use_c=True)
    new_center = dtw_barycenter.dba_loop(series, center, max_it=10, thr=0.0001, use_c=True)


**Example with differencing**:

For the Trace example above, the clustering is not perfect because the different
series have slightly different baselines that cannot be corrected with
normalization. This causes an accumulated error that is larger than the
subtle sine wave in one of the types of series. A possible solution is to
apply differencing on the signals to focus on the changes in the series.
Additionally, we also apply a low-pass filter the avoid accumulation of
noise.

::

    series = dtaidistance.preprocessing.differencing(series, smooth=0.1)
    model = KMeans(k=4, max_it=10, max_dba_it=10, dists_options={"window": 40})
    cluster_idx, performed_it = model.fit(series, use_c=True, use_parallel=False)


.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/kmeans_differencing.png?v=1
   :alt: KMeans clustering with differencing and low-pass filter


K-Medoids clustering
~~~~~~~~~~~~~~~~~~~~

The distance matrix can also be used for k-medoid time series clustering.
The ``kmedoids`` class from the `pyclustering <https://pyclustering.github.io>`_ package supports
a distance matrix as input. It is wrapped in the ``dtaidistance.clustering.medoids.KMedoids``
class.


::

    from dtaidistance import dtw, clustering
    s = np.array([
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [1., 2, 0, 0, 0, 0, 0, 1, 1]])

    model = clustering.KMedoids(dtw.distance_matrix_fast, {}, k=3)
    cluster_idx = model.fit(s)
    model.plot("kmedoids.png")


.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/kmedoids.png?v=1
   :alt: KMedoids clustering


Active semi-supervised clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended method for perform active semi-supervised clustering using
DTAIDistance is to use the COBRAS for time series clustering: https://github.com/ML-KULeuven/cobras.
COBRAS is a library for semi-supervised time series clustering using pairwise constraints,
which natively supports both dtaidistance.dtw and kshape.
