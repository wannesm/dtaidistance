import math
import pytest
import os
import numpy as np
from dtaidistance import dtw, dtw_c
from dtaidistance import dtw_visualisation as dtwvis


def test_normalize():
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
    s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
    r, path = dtw.warp(s1, s2)
    # dtwvis.plot_warp(s1, s2, r, path, filename=os.path.expanduser("~/Desktop/test_normalize1.png")
    r_c = np.array([0., 1., 2., 2., 1., 0.5, 0., 0., 2., 1., 0., 0., 0.])
    # path = dtw.warping_path(s1, s2, psi=2)
    # dtwvis.plot_warping(s1, s2, path, filename=os.path.expanduser("~/Desktop/test_normalize2.png"))
    np.testing.assert_almost_equal(r, r_c, decimal=4)


def test_psi_dtw_1a():
    x = np.arange(0, 20, .5)
    s1 = np.sin(x)
    s2 = np.sin(x - 1)
    d, paths = dtw.warping_paths(s1, s2, psi=2)
    print(paths[:5,:5])
    # path = dtw.warping_path(s1, s2, psi=2)
    # dtwvis.plot_warping(s1, s2, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_1_1.png"))
    # path = dtw.best_path(paths)
    # dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_1_2.png"))
    np.testing.assert_equal(d, 0.0)


def test_psi_dtw_1b():
    x = np.arange(0, 20, .5)
    s1 = np.sin(x)
    s2 = np.sin(x - 1)
    d = dtw.distance(s1, s2, psi=2)
    np.testing.assert_equal(d, 0.0)


def test_psi_dtw_1c():
    x = np.arange(0, 20, .5)
    s1 = np.sin(x)
    s2 = np.sin(x - 1)
    d = dtw_c.distance_nogil(s1, s2, psi=2)
    np.testing.assert_equal(d, 0.0)


def test_psi_dtw_2a():
    x = np.arange(0, 20, .5)
    s1 = np.sin(x - 1)
    s2 = np.sin(x)
    d, paths = dtw.warping_paths(s1, s2, psi=2, window=3)
    # path = dtw.warping_path(s1, s2, psi=2)
    # dtwvis.plot_warping(s1, s2, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_2_1.png"))
    # path = dtw.best_path(paths)
    # dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_2_2.png"))
    np.testing.assert_equal(d, 0.0)


def test_psi_dtw_2b():
    x = np.arange(0, 20, .5)
    s1 = np.sin(x - 1)
    s2 = np.sin(x)
    d = dtw.distance(s1, s2, psi=2, window=3)
    np.testing.assert_equal(d, 0.0)


def test_psi_dtw_2c():
    x = np.arange(0, 20, .5)
    s1 = np.sin(x - 1)
    s2 = np.sin(x)
    d = dtw_c.distance_nogil(s1, s2, psi=2, window=3)
    np.testing.assert_equal(d, 0.0)


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=120)
    # test_normalize()
    # test_psi_dtw_1a()
    # test_psi_dtw_1b()
    test_psi_dtw_1c()
    # test_psi_dtw_2a()
    # test_psi_dtw_2b()

