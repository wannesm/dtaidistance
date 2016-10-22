import numpy as np
from dtaidistance import dtw, dtw_c


def test_distance1_a():
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    d1 = dtw.distance(s1, s2)
    assert d1 == 2


def test_distance1_b():
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    d2, _ = dtw.distances(s1, s2)
    assert d2 == 2


def test_distance1_c():
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    d3 = dtw_c.distance(np.array(s1, dtype=np.float64), np.array(s2, dtype=np.float64))
    assert(d3) == 2


def test_distance1_d():
    s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.float64)
    s2 = np.array([0, 1, 2, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    d = dtw_c.distance_nogil(s1, s2)
    assert(d) == 2


def test_distance_matrix1_a():
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0],
         [0, 1, 2, 0, 0, 0, 0, 0, 0]]
    s = [np.array(si, dtype=np.float64) for si in s]
    m1 = dtw.distance_matrix(s, parallel=False, use_c=False)
    assert m1[0,1] == 2


def test_distance_matrix1_b():
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0],
         [0, 1, 2, 0, 0, 0, 0, 0, 0]]
    s = [np.array(si, dtype=np.float64) for si in s]
    m2 = dtw.distance_matrix(s, parallel=True, use_c=False)
    assert m2[0, 1] == 2


def test_distance_matrix1_c():
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0],
         [0, 1, 2, 0, 0, 0, 0, 0, 0]]
    s = [np.array(si, dtype=np.float64) for si in s]
    m3 = dtw.distance_matrix(s, parallel=False, use_c=True)
    assert m3[0, 1] == 2


def test_distance_matrix1_d():
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0],
         [0, 1, 2, 0, 0, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0, 1]]
    s = [np.array(si, dtype=np.float64) for si in s]
    m = dtw_c.distance_matrix_nogil(s)
    assert m[0, 1] == 2


def test_distance_matrix1_e():
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0],
         [0, 1, 2, 0, 0, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0, 1]]
    s = [np.array(si, dtype=np.float64) for si in s]
    m = dtw_c.distance_matrix_nogil_p(s)
    assert m[0, 1] == 2

test_distance_matrix1_e()
