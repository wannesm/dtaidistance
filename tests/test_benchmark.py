import numpy as np
from dtaidistance import dtw, clustering
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
def test_distance1_c_numpy(benchmark):
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0]*n)
    s2 = np.array([0., 1, 2, 0, 0, 0, 0, 0, 0]*n)

    def d():
        return dtw.distance_fast(s1, s2)

    assert benchmark(d) == math.sqrt(2*n)


@pytest.mark.benchmark(group="distance1")
def test_distance1_c_array(benchmark):
    s1 = array.array('d', [0., 0, 1, 2, 1, 0, 1, 0, 0]*n)
    s2 = array.array('d', [0., 1, 2, 0, 0, 0, 0, 0, 0]*n)

    def d():
        return dtw.distance_fast(s1, s2)

    assert benchmark(d) == math.sqrt(2*n)


@pytest.mark.benchmark(group="distance1")
def test_distance1_c_array_prune(benchmark):
    s1 = array.array('d', [0., 0, 1, 2, 1, 0, 1, 0, 0]*n)
    s2 = array.array('d', [0., 1, 2, 0, 0, 0, 0, 0, 0]*n)

    def d():
        return dtw.distance_fast(s1, s2, use_pruning=True)

    assert benchmark(d) == math.sqrt(2*n)


# --- DISTANCE MATRIX 1 ---


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_serial_python(benchmark):
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0, 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1, 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=False, use_c=False, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n)


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_parallel_python(benchmark):
    s = [[0, 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0, 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1, 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=True, use_c=False, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n)


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_parallel_mp_c(benchmark):
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0., 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1., 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=True, use_c=True, use_mp=True, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n)


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_serial_c(benchmark):
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0]*n,
         [0., 1, 2, 0, 0, 0, 0, 0, 0]*n,
         [1., 2, 0, 0, 0, 0, 0, 1]*n]*nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=False, use_c=True, compact=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n)


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_serial_c_pruned(benchmark):
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0]*n,
         [0., 1, 2, 0, 0, 0, 0, 0, 0]*n,
         [1., 2, 0, 0, 0, 0, 0, 1]*n]*nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=False, use_c=True, compact=True, use_pruning=True)

    m = benchmark(d)
    assert m[0] == math.sqrt(2*n)


@pytest.mark.benchmark(group="matrix1")
def test_distance_matrix1_parallel_c(benchmark):
    s = [[0., 0, 1, 2, 1, 0, 1, 0, 0] * n,
         [0., 1, 2, 0, 0, 0, 0, 0, 0] * n,
         [1., 2, 0, 0, 0, 0, 0, 1] * n] * nn
    s = [np.array(si) for si in s]

    def d():
        return dtw.distance_matrix(s, parallel=True, use_c=True, compact=True)

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
    # test_cluster1_linkage(lambda x: x())
    test_distance_matrix1_serial_python(lambda x: x())
