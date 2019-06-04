import math
import pytest
import numpy as np
from dtaidistance import dtw, dtw_c


def test_expected_length1():
    block = ((1, 4), (3, 5))
    length = dtw._distance_matrix_length(block, 6)
    assert length == 5


def test_expected_length2():
    block = ((1, 3), (3, 5))
    length = dtw._distance_matrix_length(block, 6)
    assert length == 4


def test_expected_length3():
    block = ((1, 4), (3, 5))
    length = dtw._distance_matrix_length(block, 6)
    assert length == 5


def test_expected_length4():
    block = ((0, 6), (0, 6))
    length = dtw._distance_matrix_length(block, 6)
    assert length == int(6 * (6 - 1) / 2)


def test_condensed_index1():
    """
         0    1    2    3    4    5
      +-----------------------------+
    0 | -- |  0 |  1 |  2 |  3 |  4 |
    1 | -- | -- |  5 |  6 |  7 |  8 |
    2 | -- | -- | -- |  9 | 10 | 11 |
    3 | -- | -- | -- | -- | 12 | 13 |
    4 | -- | -- | -- | -- | -- | 14 |
      +-----------------------------+

    """
    assert dtw.distance_array_index(3, 2, 6) == 9
    assert dtw.distance_array_index(2, 3, 6) == 9
    assert dtw.distance_array_index(1, 5, 6) == 8


def test_distance1_a():
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    d1 = dtw.distance(s1, s2)
    assert d1 == pytest.approx(math.sqrt(2))


def test_distance1_b():
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    d2, _ = dtw.warping_paths(s1, s2)
    assert d2 == pytest.approx(math.sqrt(2))


def test_distance1_d():
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = np.array([0., 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw_c.distance_nogil(s1, s2)
    assert(d) == pytest.approx(math.sqrt(2))


def test_distance1_c():
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = np.array([0, 1, 2, 0, 0, 0, 0, 0, 0], dtype=np.double)
    d3 = dtw_c.distance(s1, s2)
    assert(d3) == pytest.approx(math.sqrt(2))


def test_distance_matrix1_a():
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0],
         [0, 1, 2, 0, 0, 0, 0, 0, 0]]
    s = [np.array(si) for si in s]
    m1 = dtw.distance_matrix(s, parallel=False, use_c=False)
    assert m1[0,1] == pytest.approx(math.sqrt(2))


def test_distance_matrix1_b():
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0],
         [0, 1, 2, 0, 0, 0, 0, 0, 0]]
    s = [np.array(si) for si in s]
    m2 = dtw.distance_matrix(s, parallel=True, use_c=False)
    assert m2[0, 1] == pytest.approx(math.sqrt(2))


def test_distance_matrix1_c():
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0]]
    s = [np.array(si) for si in s]
    m3 = dtw.distance_matrix(s, parallel=False, use_c=True)
    assert m3[0, 1] == pytest.approx(math.sqrt(2))


def test_distance_matrix1_d():
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1]]
    s = [np.array(si) for si in s]
    m = dtw_c.distance_matrix_nogil(s)
    m = dtw.distances_array_to_matrix(m, len(s))
    assert m[0, 1] == pytest.approx(math.sqrt(2))


def test_distance_matrix1_e():
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1]]
    s = [np.array(si) for si in s]
    m = dtw_c.distance_matrix_nogil(s, is_parallel=True)
    m = dtw.distances_array_to_matrix(m, len(s))
    print(m)
    assert m[0, 1] == pytest.approx(math.sqrt(2))


def test_distance_matrix2_e():
    n = 1
    nn = 1
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0., 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1., 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]
    m1 = dtw_c.distance_matrix_nogil(s, is_parallel=True)
    m1 = dtw.distances_array_to_matrix(m1, len(s))
    m2 = dtw.distance_matrix(s, parallel=True, use_c=True, use_nogil=True)
    assert m1[0, 1] == math.sqrt(2) * n, "m1[0,1]={} != {}".format(m1[0, 1], math.sqrt(2) * n)
    assert m2[0, 1] == math.sqrt(2) * n, "m2[0,1]={} != {}".format(m2[0, 1], math.sqrt(2) * n)


def run_distance_matrix_block(parallel=False, use_c=False, use_nogil=False):
    # print(parallel, use_c, use_nogil)
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]]
    s = np.array(s)
    m = dtw.distance_matrix(s, block=((1, 4), (3, 5)), parallel=parallel, use_c=use_c, use_nogil=use_nogil)
    print(m)
    assert m[1, 3] == pytest.approx(math.sqrt(2))
    assert np.isinf(m[1, 2])


def test_distance_matrix_block():
    for parallel in [False, True]:
        for use_c in [False,True]:
            for use_nogil in [False, True]:
                run_distance_matrix_block(parallel=parallel, use_c=use_c, use_nogil=use_nogil)


if __name__ == "__main__":
    # test_distance1_a()
    # test_distance_matrix2_e()
    # run_distance_matrix_block(parallel=True, use_c=True, use_nogil=False)
    # test_expected_length1()
    test_condensed_index1()
