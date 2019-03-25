import math
import pytest
import numpy as np
from dtaidistance import dtw, dtw_ndim, dtw_ndim_visualisation as dtwvis


def test_distance1_a():
    s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1],  [0, 0]], dtype=np.double)
    s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
    d1 = dtw_ndim.distance(s1, s2)
    d1p, paths = dtw_ndim.warping_paths(s1, s2)
    assert d1 == pytest.approx(d1p)


def test_visualisation_a():
    s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1], [0, 0]], dtype=np.double)
    s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
    d1p, paths = dtw_ndim.warping_paths(s1, s2)
    path = dtw.best_path(paths)
    fig, ax = dtwvis.plot_warping(s1, s2, path)
    fig.show()

def test_visualisation_b():
    s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1], [0, 0]], dtype=np.double)
    s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
    d1p, paths = dtw_ndim.warping_paths(s1, s2)
    path = dtw.best_path(paths)
    fig, ax = dtwvis.plot_warpingpaths(s2, s1, paths, path=path)
    fig.show()


if __name__ == "__main__":
    test_distance1_a()
    test_visualisation_a()
    test_visualisation_b()
