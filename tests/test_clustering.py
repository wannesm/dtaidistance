import os
import math
import tempfile
import pytest
import numpy as np
from dtaidistance import dtw, clustering


def test_clustering():
    s = np.array([
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]])

    def test_hook(from_idx, to_idx, distance):
        assert (from_idx, to_idx) in [(3, 0), (4, 1), (5, 2), (1, 0)]
    model = clustering.Hierarchical(dtw.distance_matrix_fast, {}, 2, merge_hook=test_hook,
                                    show_progress=False)
    cluster_idx = model.fit(s)
    assert cluster_idx[0] == {0, 1, 3, 4}
    assert cluster_idx[2] == {2, 5}


def test_clustering_tree(directory=None):
    s = np.array([
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]])

    def test_hook(from_idx, to_idx, distance):
        assert (from_idx, to_idx) in [(3, 0), (4, 1), (5, 2), (6, 2), (1, 0), (2, 0)]
    model = clustering.Hierarchical(dtw.distance_matrix_fast, {}, merge_hook=test_hook,
                                    show_progress=False)
    modelw = clustering.HierarchicalTree(model)
    cluster_idx = modelw.fit(s)
    assert cluster_idx[0] == {0, 1, 2, 3, 4, 5, 6}

    if directory:
        hierarchy_fn = os.path.join(directory, "hierarchy.png")
        graphviz_fn = os.path.join(directory, "hierarchy.dot")
    else:
        file = tempfile.NamedTemporaryFile()
        hierarchy_fn = file.name + "_hierarchy.png"
        graphviz_fn = file.name + "_hierarchy.dot"
    modelw.plot(hierarchy_fn)
    print("Figure saved to", hierarchy_fn)
    with open(graphviz_fn, "w") as ofile:
        print(modelw.to_dot(), file=ofile)
    print("Dot saved to", graphviz_fn)


def test_linkage_tree(directory=None):
    s = np.array([
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]])

    model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
    cluster_idx = model.fit(s)

    if directory:
        hierarchy_fn = os.path.join(directory, "hierarchy.png")
        graphviz_fn = os.path.join(directory, "hierarchy.dot")
    else:
        file = tempfile.NamedTemporaryFile()
        hierarchy_fn = file.name + "_hierarchy.png"
        graphviz_fn = file.name + "_hierarchy.dot"
    model.plot(hierarchy_fn)
    print("Figure saved to", hierarchy_fn)
    with open(graphviz_fn, "w") as ofile:
        print(model.to_dot(), file=ofile)
    print("Dot saved to", graphviz_fn)


if __name__ == "__main__":
    # test_clustering_tree(directory="/Users/wannes/Desktop/")
    test_linkage_tree(directory="/Users/wannes/Desktop/")
