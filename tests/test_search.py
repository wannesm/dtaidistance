import logging
import os
from itertools import product

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from dtaidistance import dtw, dtw_search, util_numpy

logger = logging.getLogger("be.kuleuven.dtai.distance")
directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@pytest.fixture(scope="session")
def data_trace():
    scaler = StandardScaler()

    rsrc_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'Trace_TRAIN.txt')
    data = np.loadtxt(rsrc_fn)
    series = scaler.fit_transform(data[:,1:len(data[0])])

    return series

@numpyonly
@pytest.mark.parametrize("use_c", [True, False])
def test_keogh_envelope(use_c):
    with util_numpy.test_uses_numpy() as np:
        ts1 = np.array([1, 2, 3, 2, 1], dtype=np.double)
        (L, U) = dtw_search.lb_keogh_envelope(ts1, window=1, use_c=use_c)
        assert np.array_equal(L, np.array([1., 1., 2., 1., 1.]))
        assert np.array_equal(U, np.array([2., 3., 3., 3., 2.]))


@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True), (True, False)])
def test_keogh_envelope_iterator(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
        ts1 = np.array([[1, 2, 3, 2, 1]], dtype=np.double)
        [(L, U)] = dtw_search.lb_keogh_envelope(ts1, window=1, use_c=use_c, parallel=parallel)
        assert np.array_equal(L, np.array([1., 1., 2., 1., 1.]))
        assert np.array_equal(U, np.array([2., 3., 3., 3., 2.]))


@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True)])
def test_keogh_distance_equal_lenghts(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
      ts1 = np.array([1, 2, 3, 2, 1], dtype=np.double)
      ts2 = np.array([0, 0, 0, 0, 0], dtype=np.double)
      d = dtw_search.lb_keogh(ts2, ts1, use_c=use_c, parallel=parallel)
      assert d == 5


@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True)])
def test_keogh_distance_equal_lenghts_iterator_both(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
        ts1 = np.array([[1, 2, 3, 2, 1], [2, 2, 2, 2, 1]], dtype=np.double)
        ts2 = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=np.double)
        d = dtw_search.lb_keogh(ts2, ts1, None, use_c=use_c, parallel=parallel)
        assert np.array_equal(d, np.array([[5., 5.], [0, 0]]))


@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True)])
def test_keogh_distance_equal_lenghts_iterator_first(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
      ts1 = np.array([[1, 2, 3, 2, 1]], dtype=np.double)
      ts2 = np.array([0, 0, 0, 0, 0], dtype=np.double)
      d = dtw_search.lb_keogh(ts2, ts1, None, use_c=use_c, parallel=parallel)
      assert d == [5]


@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True)])
def test_keogh_distance_equal_lenghts_iterator_second(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
      ts1 = np.array([1, 2, 3, 2, 1], dtype=np.double)
      ts2 = np.array([[0, 0, 0, 0, 0]], dtype=np.double)
      d = dtw_search.lb_keogh(ts2, ts1, None, use_c=use_c, parallel=parallel)
      assert d == [5]

@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True)])
def test_keogh_distance_unequal_lenghts(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
      ts1 = np.array([1, 3, 2, 0 ,3, 2, 1, 2], dtype=np.double)
      ts2 = np.array([0, 0, 0, 0, 0], dtype=np.double)
      d = dtw_search.lb_keogh(ts2, ts1, window=1, use_c=use_c, parallel=parallel)
      assert d == 1


@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True)])
def test_keogh_distance_unequal_lenghts_iterator(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
      ts1 = np.array([[1, 3, 2, 0 ,3, 2, 1, 2]], dtype=np.double)
      ts2 = np.array([[0, 0, 0, 0, 0]], dtype=np.double)
      d = dtw_search.lb_keogh(ts2, ts1, window=1, use_c=use_c, parallel=parallel)
      assert d == [[1]]


@numpyonly
@pytest.mark.parametrize("use_c, parallel", [(False, False), (True, False), (True, True)])
def test_nn_keogh(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:

        X = np.array([
            [1, 5, 6, 1, 1], 
            [1, 2, 7, 2, 1],
            [25, 22, 15, 41, 21]], dtype=np.double)
        X2 = np.array([
            [1, 5, 6, 1, 1], 
            [25, 2, 15, 41, 21],
            [25, 22, 15, 41, 21], 
            [1, 2, 7, 2, 1]], dtype=np.double)

        Y = dtw_search.nearest_neighbour_lb_keogh(X, X2, None, dist_params={'window': 2, 'psi': 0}, use_c=use_c, parallel=parallel, use_mp=False)
        assert np.array_equal(Y, np.array([0,2,2,1], dtype=np.int))
        envelopes = dtw_search.lb_keogh_envelope(X2, window=2, use_c=use_c, parallel=parallel)
        Y = dtw_search.nearest_neighbour_lb_keogh(X, X2, envelopes, dist_params={'window': 2, 'psi': 0}, use_c=use_c, parallel=parallel, use_mp=False)
        assert np.array_equal(Y, np.array([0,2,2,1], dtype=np.int))


@numpyonly
@pytest.mark.parametrize("use_c,use_ucr", [(True, True), (True, False), (False, False)])
def test_nn_subsequence(use_c, use_ucr):
    with util_numpy.test_uses_numpy() as np:

        X = np.array([
            1, 5, 6, 1, 1, 
            1, 2, 7, 2, 1,
            25, 22, 15, 41, 21], dtype=np.double)
        X2 = np.array([
            [1, 5, 6, 1, 1], 
            [25, 2, 15, 41, 21],
            [25, 22, 15, 41, 21], 
            [1, 2, 7, 2, 1]], dtype=np.double)

        Y = dtw_search.nearest_neighbour_lb_keogh(X, X2, None, dist_params={'window': 2, 'psi': 0}, use_c=use_c, use_ucr=use_ucr)
        assert np.array_equal(Y, np.array([0,10,10,5], dtype=np.int))
        envelopes = dtw_search.lb_keogh_envelope(X2, window=2, use_c=use_c, parallel=True)
        Y = dtw_search.nearest_neighbour_lb_keogh(X, X2, envelopes, dist_params={'window': 2, 'psi': 0}, use_c=use_c, use_ucr=use_ucr)
        assert np.array_equal(Y, np.array([0,10,10,5], dtype=np.int))



@numpyonly
def test_nn_ucr():
    with util_numpy.test_uses_numpy() as np:

        X = np.array([
            [1, 5, 6, 1, 1], 
            [1, 2, 7, 2, 1],
            [25, 22, 15, 41, 21]], dtype=np.double)
        X2 = np.array([
            [1, 5, 6, 1, 1], 
            [25, 2, 15, 41, 21],
            [25, 22, 15, 41, 21], 
            [1, 2, 7, 2, 1]], dtype=np.double)

        Y = dtw_search.nearest_neighbour_lb_keogh(X, X2, None, dist_params={'window': 10, 'psi': 0}, use_c=True, use_ucr=True)

        assert np.array_equal(Y, np.array([0,2,2,1], dtype=np.int))


@pytest.mark.benchmark(group="1NN")
def test_nn_dtw_trace(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)

    def d():
        distmat = dtw.distance_matrix_fast(np.concatenate((query, space), axis=0), block=((0,2),(2,data_trace.shape[0])), **{'window': 10, 'psi': 0})
        np.fill_diagonal(distmat, np.inf)
        q1 = np.argmin(distmat[0,:]) - 2
        q2 = np.argmin(distmat[1,:]) - 2
        return [q1, q2]

    Y = benchmark(d)
    assert np.array_equal(Y, [29, 26])


@pytest.mark.benchmark(group="1NN")
def test_nn_keogh_trace_python(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)
    params = {'window': 10, 'psi': 0}

    envelopes = dtw_search.lb_keogh_envelope(query, window=params['window'], use_c=False, parallel=True)
    def d():
        Y = dtw_search.nearest_neighbour_lb_keogh(space, query, envelopes, dist_params=params, use_c=False, parallel=True)
        return Y

    Y = benchmark(d)
    assert np.array_equal(Y, [29,26])


@pytest.mark.benchmark(group="1NN")
def test_nn_keogh_trace_C(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)
    params = {'window': 10, 'psi': 0}

    envelopes = dtw_search.lb_keogh_envelope(query, window=params['window'], use_c=True, parallel=True)
    def d():
        Y = dtw_search.nearest_neighbour_lb_keogh(space, query, envelopes, dist_params=params, use_c=True, parallel=True)
        return Y

    Y = benchmark(d)
    assert np.array_equal(Y, [29,26])


@pytest.mark.benchmark(group="1NN Subsequence")
def test_nn_dtw_subsequence_trace(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double).ravel()
    rolling_space = list()
    # FIXME: This can probably be made a lot more efficient
    l = query.shape[1]
    for i in range(len(space)-l+1):
        rolling_space.append(space[i:i+l])
    rolling_space= np.array(rolling_space)

    def d():
        distmat = dtw.distance_matrix_fast(np.concatenate((query, rolling_space), axis=0), block=((0,2),(2,rolling_space.shape[0]+2)), **{'window': 10, 'psi': 0})
        np.fill_diagonal(distmat, np.inf)
        q1 = np.argmin(distmat[0,:]) - 2
        q2 = np.argmin(distmat[1,:]) - 2
        return [q1, q2]

    Y = benchmark(d)
    assert np.array_equal(Y, [7976, 7149])


@pytest.mark.benchmark(group="1NN Subsequence")
def test_nn_ucr_subsequence_trace(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)

    def d():
        Y = dtw_search.nearest_neighbour_lb_keogh(space.ravel(), query, None, dist_params={'window': 10, 'psi': 0}, use_c=True, use_ucr=True, parallel=True)
        return Y

    Y = benchmark(d)
    assert np.array_equal(Y, [7976, 7149])


@pytest.mark.benchmark(group="1NN Subsequence")
def test_nn_keogh_subsequence_trace(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)

    def d():
        Y = dtw_search.nearest_neighbour_lb_keogh(space.ravel(), query, None, dist_params={'window': 10, 'psi': 0}, use_c=True, use_ucr=False, parallel=True)
        return Y

    Y = benchmark(d)
    assert np.array_equal(Y, [7976, 7149])


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # test_keogh_envelope(True)
    # test_keogh_nn(True)

