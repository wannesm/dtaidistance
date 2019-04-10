import pytest
import numpy as np
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from dtaidistance import dtw, dtw_c


def test_numpymatrix():
    """Passing a matrix instead of a list failed because the array is now a
    view instead of the original data structure."""
    s = np.array([
        [0., 0, 1, 2, 1, 0, 1, 0, 0],
        [0., 1, 2, 0, 0, 0, 0, 0, 0],
        [1., 2, 0, 0, 0, 0, 0, 1, 0]])
    m = dtw_c.distance_matrix_nogil(s)
    m = dtw.distances_array_to_matrix(m, len(s))
    m2 = dtw.distance_matrix(s)
    correct = np.array([
        [np.inf, 1.41421356, 1.73205081],
        [np.inf, np.inf,     1.41421356],
        [np.inf, np.inf,     np.inf]])
    assert m[0, 1] == pytest.approx(math.sqrt(2))
    assert m2[0, 1] == pytest.approx(math.sqrt(2))
    np.testing.assert_almost_equal(correct, m, decimal=4)
    np.testing.assert_almost_equal(correct, m2, decimal=4)


def test_numpymatrix_compact():
    """Passing a matrix instead of a list failed because the array is now a
    view instead of the original data structure."""
    s = np.array([
        [0., 0, 1, 2, 1, 0, 1, 0, 0],
        [0., 1, 2, 0, 0, 0, 0, 0, 0],
        [1., 2, 0, 0, 0, 0, 0, 1, 0]])
    m = dtw_c.distance_matrix_nogil(s)
    m2 = dtw.distance_matrix(s, compact=True)
    correct = np.array([1.41421356, 1.73205081, 1.41421356])
    assert m[0] == pytest.approx(math.sqrt(2))
    assert m2[0] == pytest.approx(math.sqrt(2))
    np.testing.assert_almost_equal(correct, m, decimal=4)
    np.testing.assert_almost_equal(correct, m2, decimal=4)


def test_numpymatrix_transpose():
    """Passing a matrix instead of a list failed because the array is now a
    view instead of the original data structure."""
    s = np.array([
        [0., 0., 1.,],
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 0]
    ]).T
    m = dtw_c.distance_matrix_nogil(s)
    m = dtw.distances_array_to_matrix(m, len(s))
    m2 = dtw.distance_matrix(s)
    correct = np.array([
        [np.inf, 1.41421356, 1.73205081],
        [np.inf, np.inf,     1.41421356],
        [np.inf, np.inf,     np.inf]])
    assert m[0, 1] == pytest.approx(math.sqrt(2))
    assert m2[0, 1] == pytest.approx(math.sqrt(2))
    np.testing.assert_almost_equal(correct, m, decimal=4)
    np.testing.assert_almost_equal(correct, m2, decimal=4)


if __name__ == "__main__":
    test_numpymatrix()
    test_numpymatrix_transpose()
