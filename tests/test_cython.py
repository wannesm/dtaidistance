import pytest
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from dtaidistance import dtw, util_numpy


numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_numpymatrix():
    """Passing a matrix instead of a list failed because the array is now a
    view instead of the original data structure."""
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
            [0., 0, 1, 2, 1, 0, 1, 0, 0],
            [0., 1, 2, 0, 0, 0, 0, 0, 0],
            [1., 2, 0, 0, 0, 0, 0, 1, 0]])
        m = dtw.distance_matrix_fast(s, only_triu=True)
        m2 = dtw.distance_matrix(s, only_triu=True)
        correct = np.array([
            [np.inf, 1.41421356, 1.73205081],
            [np.inf, np.inf,     1.41421356],
            [np.inf, np.inf,     np.inf]])
        assert m[0, 1] == pytest.approx(math.sqrt(2))
        assert m2[0, 1] == pytest.approx(math.sqrt(2))
        np.testing.assert_almost_equal(correct, m, decimal=4)
        np.testing.assert_almost_equal(correct, m2, decimal=4)


@numpyonly
def test_numpymatrix_compact():
    """Passing a matrix instead of a list failed because the array is now a
    view instead of the original data structure."""
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
            [0., 0, 1, 2, 1, 0, 1, 0, 0],
            [0., 1, 2, 0, 0, 0, 0, 0, 0],
            [1., 2, 0, 0, 0, 0, 0, 1, 0]])
        m = dtw.distance_matrix_fast(s, compact=True)
        m2 = dtw.distance_matrix(s, compact=True)
        correct = np.array([1.41421356, 1.73205081, 1.41421356])
        assert m[0] == pytest.approx(math.sqrt(2))
        assert m2[0] == pytest.approx(math.sqrt(2))
        np.testing.assert_almost_equal(correct, m, decimal=4)
        np.testing.assert_almost_equal(correct, m2, decimal=4)


@numpyonly
def test_numpymatrix_transpose():
    """Passing a matrix instead of a list failed because the array is now a
    view instead of the original data structure."""
    with util_numpy.test_uses_numpy() as np:
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
        m = dtw.distance_matrix_fast(s, only_triu=True)
        m2 = dtw.distance_matrix(s, only_triu=True)
        correct = np.array([
            [np.inf, 1.41421356, 1.73205081],
            [np.inf, np.inf,     1.41421356],
            [np.inf, np.inf,     np.inf]])
        assert m[0, 1] == pytest.approx(math.sqrt(2))
        assert m2[0, 1] == pytest.approx(math.sqrt(2))
        np.testing.assert_almost_equal(correct, m, decimal=4)
        np.testing.assert_almost_equal(correct, m2, decimal=4)


# def test_negativedimensions():
#     """Failed for sizes of (62706, 104): ValueError: negative dimensions are not allowed
#     """
#     s = np.array([[ 0.,  0.,  0.,  0.,  2.,  4.,  5., 10., 11., 10., 10.,  6.,  3.,
#          2.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  0.,  2.,
#          4.,  4.,  5.,  5.,  3.,  4.,  4.,  2.,  2.,  2.,  3.,  3.,  4.,
#          4.,  4.,  4.,  2.,  2.,  3.,  3.,  7.,  6.,  4.,  4.,  8.,  8.,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
#        [ 1., np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN],
#        [ 3.,  4., 11., 13., 12., 10.,  3.,  1.,  5.,  3., 14., 20., 37.,
#         40., 38., 31., 18., 10.,  9., 12.,  8., 16., 15., 19., 27., 26.,
#         30., 23., 18., 15., 13., 25., 28., 25., 27., 15.,  8., 11., 13.,
#         23., 18., 12.,  6., np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
#         np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]])
#
#     window = 1
#     dists = dtw.distance_matrix_fast(s, window=int(0.01 * window * s.shape[1]))
#     print(dists)
#
#     dists = dtw.distance_matrix(s, window=int(0.01 * window * s.shape[1]))
#     print(dists)
#
#     wp = dtw.warping_paths(s[0], s[2])
#     print(wp)


@numpyonly
def test_negativedimensions2():
    """
    For length 62706, the computation (len_cur * (len_cur - 1)) is a number that is too large.

    For length 62706 the length calculation for the distance matrix returned 1965958512 but should return 1965989865
    Without parenthesis len_cur * (len_cur - 1) / 2 returns -181493783
    Incorrect behaviour can be simulated using fixedint.Int32(62706)
    Problem is with dividing an uneven number to optimize the computation and require smaller number.
    This dividing the uneven number should and can be avoided.
    """
    with util_numpy.test_uses_numpy() as np:
        # s = np.full((62706, 104), 1.0)
        s = np.full((10, 104), 1.0)
        window = 1
        dists = dtw.distance_matrix_fast(s, window=int(0.01 * window * s.shape[1]))
        print(dists)


@numpyonly
def test_overflowdistance():
    with util_numpy.test_uses_numpy() as np:
        maxvalthirtytwobit = 2147483647
        s = np.array([
            [maxvalthirtytwobit, maxvalthirtytwobit, 1, 2, 1, 0, 1, 0, 0],
            [1., 2, 0, 0, 0, 0, 0, 1, 0]])
        d1 = dtw.distance(s[0], s[1], use_c=False)
        d2 = dtw.distance(s[0], s[1], use_c=True)
        print(d1)
        print(d2)
        # m = dtw_c.distance_matrix_nogil(s)
        # m2 = dtw.distance_matrix(s, compact=True)
        # print(m)
        # print(m2)
        # correct = np.array([1.41421356, 1.73205081, 1.41421356])
        # assert m[0] == pytest.approx(math.sqrt(2))
        # assert m2[0] == pytest.approx(math.sqrt(2))
        # np.testing.assert_almost_equal(correct, m, decimal=4)
        # np.testing.assert_almost_equal(correct, m2, decimal=4)


if __name__ == "__main__":
    # test_numpymatrix()
    # test_numpymatrix_transpose()
    # test_negativedimensions()
    # test_negativedimensions2()
    test_overflowdistance()

