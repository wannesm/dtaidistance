import logging
import sys
import pytest
import numpy as np
from dtaidistance import dtw, dtw_ndim, dtw_ndim_visualisation as dtwvis


logger = logging.getLogger("be.kuleuven.dtai.distance")


def test_distance1_a():
    s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1],  [0, 0]], dtype=np.double)
    s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
    d1 = dtw_ndim.distance(s1, s2)
    d1p, paths = dtw_ndim.warping_paths(s1, s2)
    print(d1, d1p)
    assert d1 == pytest.approx(d1p)


def test_distance1_b():
    s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1],  [0, 0]], dtype=np.double)
    s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
    d1 = dtw_ndim.distance_fast(s1, s2)
    print(d1)


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


def test_distances1_python():
    s = np.array(
        [[[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]],
         [[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]]])
    m = dtw_ndim.distance_matrix(s, 2, compact=True)
    assert m[0] == pytest.approx(2.44948974, abs=1e-3)
    assert m[1] == pytest.approx(3.0000)
    assert m[2] == pytest.approx(0.0000)
    assert m[3] == pytest.approx(2.4495, abs=1e-3)
    assert m[4] == pytest.approx(3.0000)


def test_distances1_fast():
    s = np.array(
        [[[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]],
         [[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]]])
    m = dtw_ndim.distance_matrix_fast(s, 2, compact=True, parallel=False)
    print(m)
    assert m[0] == pytest.approx(2.44948974, abs=1e-3)
    assert m[1] == pytest.approx(3.0000)
    assert m[2] == pytest.approx(0.0000)
    assert m[3] == pytest.approx(2.4495, abs=1e-3)
    assert m[4] == pytest.approx(3.0000)


def test_distances1_fast_parallel():
    s = np.array(
        [[[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]],
         [[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]]])
    m = dtw_ndim.distance_matrix_fast(s, 2, compact=True, parallel=True)
    # print(m)
    assert m[0] == pytest.approx(2.44948974, abs=1e-3)
    assert m[1] == pytest.approx(3.0000)
    assert m[2] == pytest.approx(0.0000)
    assert m[3] == pytest.approx(2.4495, abs=1e-3)
    assert m[4] == pytest.approx(3.0000)


def test_distances2_fast():
    s = [[[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]],
         [[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]]]
    s = [np.array(a) for a in s]
    m = dtw_ndim.distance_matrix_fast(s, 2, compact=True, parallel=False)
    print(m)
    assert m[0] == pytest.approx(2.44948974, abs=1e-3)
    assert m[1] == pytest.approx(3.0000)
    assert m[2] == pytest.approx(0.0000)
    assert m[3] == pytest.approx(2.4495, abs=1e-3)
    assert m[4] == pytest.approx(3.0000)


def test_distances2_fast_parallel():
    s = [[[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]],
         [[0., 0], [1, 2], [1, 0], [1, 0]],
         [[0., 1], [2, 0], [0, 0], [0, 0]],
         [[1., 2], [0, 0], [0, 0], [0, 1]]]
    s = [np.array(a) for a in s]
    m = dtw_ndim.distance_matrix_fast(s, 2, compact=True, parallel=True)
    print(m)
    assert m[0] == pytest.approx(2.44948974, abs=1e-3)
    assert m[1] == pytest.approx(3.0000)
    assert m[2] == pytest.approx(0.0000)
    assert m[3] == pytest.approx(2.4495, abs=1e-3)
    assert m[4] == pytest.approx(3.0000)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # test_distance1_a()
    # test_distance1_b()
    # test_visualisation_a()
    # test_visualisation_b()
    # test_distances2_fast()
    test_distances2_fast_parallel()
