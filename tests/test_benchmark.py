import numpy as np
from dtaidistance import dtw, dtw_c
import array

n=10

def test_distance1_python_compress(benchmark):
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]*n
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]*n
    def d():
        return dtw.distance(s1, s2)
    assert benchmark(d) == 2*n


def test_distance1_python_matrix(benchmark):
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]*n
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]*n
    def d():
        dd, _ = dtw.distances(s1, s2)
        return dd
    assert benchmark(d) == 2*n


def test_distance1_cpython(benchmark):
    s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0]*n, dtype=np.float64)
    s2 = np.array([0, 1, 2, 0, 0, 0, 0, 0, 0]*n, dtype=np.float64)
    def d():
        return dtw_c.distance(s1, s2)
    assert benchmark(d) == 2*n


def test_distance1_c_array(benchmark):
    s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0]*n, dtype=np.float64)
    s2 = np.array([0, 1, 2, 0, 0, 0, 0, 0, 0]*n, dtype=np.float64)
    def d():
        return dtw_c.distance_nogil(s1, s2)
    assert benchmark(d) == 2*n


def test_distance1_c_numpy(benchmark):
    s1 = array.array('d',[0, 0, 1, 2, 1, 0, 1, 0, 0]*n)
    s2 = array.array('d',[0, 1, 2, 0, 0, 0, 0, 0, 0]*n)
    def d():
        return dtw_c.distance_nogil(s1, s2)
    assert benchmark(d) == 2*n
