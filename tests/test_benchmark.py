import numpy as np
from dtaidistance import dtw, dtw_c, clustering
import array
import pytest
import math

n = 1
nn = 100


# --- DISTANCE 1 ---


@pytest.mark.benchmark(group="distance1")
def test_distance1_python_compress(benchmark):
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]*n
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]*n

    def d():
        return dtw.distance(s1, s2)

    assert benchmark(d) == math.sqrt(2*n)


@pytest.mark.benchmark(group="distance1")
def test_distance1_python_matrix(benchmark):
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]*n
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]*n

    def d():
        dd, _ = dtw.warping_paths(s1, s2)
        return dd

    assert benchmark(d) == math.sqrt(2*n)


@pytest.mark.benchmark(group="distance1")
def test_distance1_cpython(benchmark):
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0]*n)
    s2 = np.array([0., 1, 2, 0, 0, 0, 0, 0, 0]*n)

    def d():
        return dtw_c.distance(s1, s2)

    assert benchmark(d) == math.sqrt(2*n)


@pytest.mark.benchmark(group="distance1")
def test_distance1_c_array(benchmark):
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0]*n)
    s2 = np.array([0., 1, 2, 0, 0, 0, 0, 0, 0]*n)

    def d():
        return dtw_c.distance_nogil(s1, s2)

    assert benchmark(d) == math.sqrt(2*n)


@pytest.mark.benchmark(group="distance1")
def test_distance1_c_numpy(benchmark):
    s1 = array.array('d', [0., 0, 1, 2, 1, 0, 1, 0, 0]*n)
    s2 = array.array('d', [0., 1, 2, 0, 0, 0, 0, 0, 0]*n)

    def d():
        return dtw_c.distance_nogil(s1, s2)

    assert benchmark(d) == math.sqrt(2*n)


# --- DISTANCE MATRIX 1 ---


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_serialpython(benchmark):
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0, 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1, 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=False, use_c=False, use_nogil=False, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n)


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_parallelpython(benchmark):
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0, 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1, 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=True, use_c=False, use_nogil=False, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n)


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_serialpythonc(benchmark):
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0., 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1., 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=False, use_c=True, use_nogil=False, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2)*n


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_parallelpythonc(benchmark):
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0., 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1., 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=True, use_c=True, use_nogil=False, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n)


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_serialpurec(benchmark):
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0]*n,
         [0., 1, 2, 0, 0, 0, 0, 0, 0]*n,
         [1., 2, 0, 0, 0, 0, 0, 1]*n]*nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=False, use_c=True, use_nogil=True, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n)


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_parallelpurec(benchmark):
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0., 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1., 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=True, use_c=True, use_nogil=True, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n), "m[0,1]={} != {}".format(m[0, 1], math.sqrt(2*n))
    assert m[0] == pytest.approx(math.sqrt(2*n))


# --- CLUSTER MATRIX 1 ---

@pytest.mark.benchmark(group="cluster1")
def test_cluster1_hierarchical(benchmark):
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0, 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1, 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        c = clustering.Hierarchical(dtw.distance_matrix_fast, {})
        return c.fit(s)

    benchmark(d)


@pytest.mark.benchmark(group="cluster1")
def test_cluster1_linkage(benchmark):
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0, 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1, 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        c = clustering.LinkageTree(dtw.distance_matrix_fast, {})
        return c.fit(s)

    benchmark(d)


if __name__ == "__main__":
    # test_distance1_c_numpy(lambda x: x())
    test_cluster1_linkage(lambda x: x())
