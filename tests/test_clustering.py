import math
import pytest
import numpy as np
from dtaidistance import dtw, clustering


def test_clustering():
    s = np.matrix([
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]])

    def test_hook(from_idx, to_idx):
        assert (from_idx, to_idx) in [(3, 0), (4, 1), (5, 2), (1, 0)]
    model = clustering.Hierarchical(dtw.distance_matrix_fast, {}, 2, weights=None, merge_hook=test_hook,
                                    show_progress=False)
    cluster_idx = model.fit(s)
    assert cluster_idx[0] == {0, 1, 3, 4}
    assert cluster_idx[2] == {2, 5}


if __name__ == "__main__":
    test_clustering()
