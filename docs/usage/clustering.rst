Clustering
----------

Agglomerative clustering
~~~~~~~~~~~~~~~~~~~~~~~~

A distance matrix can be used for time series clustering. You can use
existing methods such as ``scipy.cluster.hierarchy.linkage`` or one of
two included clustering methods (the latter is a wrapper for the SciPy
linkage method).

::

    # Custom Hierarchical clustering
    model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
    # Keep track of full tree by using the HierarchicalTree wrapper class
    model2 = clustering.HierarchicalTree(model1)
    # You can also pass keyword arguments identical to instantiate a Hierarchical object
    model2 = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
    # SciPy linkage clustering
    model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
    cluster_idx = model3.fit(series)

For models that keep track of the full clustering tree
(``HierarchicalTree`` or ``LinkageTree``), the tree can be visualised:

::

    model2.plot("myplot.png")

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/hierarchy.png?v=1
   :alt: Clustering hierarchy


Active semi-supervised clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended method for perform active semi-supervised clustering using
DTAIDistance is to use the COBRAS for time series clustering: https://bitbucket.org/toon_vc/cobras_ts.
COBRAS is a library for semi-supervised time series clustering using pairwise constraints,
which natively supports both dtaidistance.dtw and kshape.
