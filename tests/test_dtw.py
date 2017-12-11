import math
import pytest
import numpy as np
from dtaidistance import dtw, dtw_c


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
    assert m[0, 1] == pytest.approx(math.sqrt(2))


def test_distance_matrix1_e():
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1]]
    s = [np.array(si) for si in s]
    m = dtw_c.distance_matrix_nogil_p(s)
    assert m[0, 1] == pytest.approx(math.sqrt(2))


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
    # print(m)
    assert m[1, 3] == pytest.approx(math.sqrt(2))
    assert np.isinf(m[1, 2])


def test_distance_matrix_block():
    for parallel in [False, True]:
        for use_c in [False,True]:
            for use_nogil in [False, True]:
                run_distance_matrix_block(parallel=parallel, use_c=use_c, use_nogil=use_nogil)


if __name__ == "__main__":
    test_distance1_a()
