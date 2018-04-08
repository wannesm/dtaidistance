Clustering
----------

A distance matrix can be used for time series clustering. You can use
existing methods such as ``scipy.cluster.hierarchy.linkage`` or one of
two included clustering methods (the latter is a wrapper for the SciPy
linkage method).

::

    # Custom Hierarchical clustering
    model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
    # Keep track of full tree
    model2 = clustering.HierarchicalTree(model)
    # SciPy linkage clustering
    model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
    cluster_idx = model3.fit(series)

For models that keep track of the full clustering tree
(``HierarchicalTree`` or ``LinkageTree``), the tree can be visualised:

::

    model.plot("myplot.png")

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/hierarchy.png?v=1
   :alt: Clustering hierarchy

   Clustering hierarchy
