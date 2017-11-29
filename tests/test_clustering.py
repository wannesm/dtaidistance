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
    model = clustering.Hierarchical(dtw.distance_matrix_fast, {}, 2, weights=None, merge_hook=test_hook,
                                    show_progress=False)
    cluster_idx = model.fit(s)
    assert cluster_idx[0] == {0, 1, 3, 4}
    assert cluster_idx[2] == {2, 5}


def test_clustering_tree():
    s = np.array([
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]])

    def test_hook(from_idx, to_idx, distance):
        assert (from_idx, to_idx) in [(3, 0), (4, 1), (5, 2), (1, 0), (2, 0)]
    model = clustering.HierarchicalTree(dtw.distance_matrix_fast, {}, weights=None, merge_hook=test_hook,
                                        show_progress=False)
    cluster_idx = model.fit(s)
    assert cluster_idx[0] == {0, 1, 2, 3, 4, 5}

    file = tempfile.NamedTemporaryFile()
    filename = file.name + ".pdf"
    model.plot(filename)
    print("Figure saved to", filename)


if __name__ == "__main__":
    test_clustering_tree()
