import logging
import os
from itertools import product

import numpy as np
import numpy.testing as npt
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


def nn_dtw(space, query, dist_params):
    distmat = dtw.distance_matrix_fast(
            np.concatenate((query, space), axis=0), 
            block=((0,query.shape[0]),(query.shape[0], space.shape[0] + query.shape[0])),
            **dist_params)
    np.fill_diagonal(distmat, np.inf)
    return np.argmin(distmat, axis=0)[:query.shape[0]] - query.shape[0]


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
      d = dtw_search.lb_keogh(ts2, ts1, window=1, use_c=use_c, parallel=parallel)
      assert d == pytest.approx(2.8284, 0.0001)


@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True)])
def test_keogh_distance_equal_lenghts_iterator_both(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
        ts1 = np.array([[1, 2, 3, 2, 1], [2, 2, 2, 2, 1]], dtype=np.double)
        ts2 = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=np.double)
        d = dtw_search.lb_keogh(ts2, ts1, None, window=1, use_c=use_c, parallel=parallel)
        npt.assert_almost_equal(d, np.array([[2.8284, 3.7416], [1, 1.7320]]), decimal=4)


@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True)])
def test_keogh_distance_equal_lenghts_iterator_first(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
      ts1 = np.array([[1, 2, 3, 2, 1]], dtype=np.double)
      ts2 = np.array([0, 0, 0, 0, 0], dtype=np.double)
      d = dtw_search.lb_keogh(ts2, ts1, None, window=1, use_c=use_c, parallel=parallel)
      npt.assert_almost_equal(d, np.array([[2.8284]]), decimal=4)


@numpyonly
@pytest.mark.parametrize("use_c,parallel", [(False, False), (True, True)])
def test_keogh_distance_equal_lenghts_iterator_second(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
      ts1 = np.array([1, 2, 3, 2, 1], dtype=np.double)
      ts2 = np.array([[0, 0, 0, 0, 0]], dtype=np.double)
      d = dtw_search.lb_keogh(ts2, ts1, None, window=1, use_c=use_c, parallel=parallel)
      npt.assert_almost_equal(d, np.array([[2.8284]]), decimal=4)


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
      npt.assert_almost_equal(d, np.array([[1]]), decimal=4)


@numpyonly
@pytest.mark.parametrize("use_c, parallel", [(False, False), (True, False), (True, True)])
def test_nn_keogh(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:

        space = np.array([
            [1, 5, 6, 1, 1], 
            [1, 2, 7, 2, 1],
            [25, 22, 15, 41, 21]], dtype=np.double)
        query = np.array([
            [1, 5, 6, 1, 1], 
            [25, 2, 15, 41, 21],
            [25, 22, 15, 41, 21], 
            [1, 2, 7, 2, 1]], dtype=np.double)
        dist_params = {'window': 2, 'psi': 0}


        truth = nn_dtw(space, query, dist_params)

        # Without precomputed envelopes
        pred = dtw_search.nearest_neighbour_lb_keogh(space, query, None, dist_params=dist_params, use_c=use_c, parallel=parallel, use_mp=False)
        npt.assert_equal(truth, pred)

        # With precomputed envelopes
        envelopes = dtw_search.lb_keogh_envelope(query, dist_params.get('window', None), use_c=use_c, parallel=parallel)
        pred = dtw_search.nearest_neighbour_lb_keogh(space, query, envelopes, dist_params, use_c=use_c, parallel=parallel, use_mp=False)
        npt.assert_equal(truth, pred)


@numpyonly
@pytest.mark.parametrize("use_c,use_ucr", [(True, True), (True, False), (False, False)])
def test_nn_subsequence(use_c, use_ucr):
    with util_numpy.test_uses_numpy() as np:

        seq = np.array([
            1, 5, 6, 1, 1, 
            1, 2, 7, 2, 1,
            25, 22, 15, 41, 21], dtype=np.double)
        subseq = np.array([
            [1, 5, 6, 1, 1], 
            [25, 2, 15, 41, 21],
            [25, 22, 15, 41, 21], 
            [1, 2, 7, 2, 1]], dtype=np.double)
        dist_params = {'window': 2, 'psi': 0}

        # Without precomputed envelopes
        pred = dtw_search.nearest_neighbour_lb_keogh(seq, subseq, None, dist_params=dist_params, use_c=use_c, use_ucr=use_ucr)
        npt.assert_equal(np.array([0,10,10,5]), pred)

        # With precomputed envelopes
        envelopes = dtw_search.lb_keogh_envelope(subseq, window=2, use_c=use_c, parallel=True)
        pred = dtw_search.nearest_neighbour_lb_keogh(seq, subseq, envelopes, dist_params=dist_params, use_c=use_c, use_ucr=use_ucr)
        npt.assert_equal(np.array([0,10,10,5]), pred)


@numpyonly
def test_nn_ucr():
    with util_numpy.test_uses_numpy() as np:

        space = np.array([
            [1, 5, 6, 1, 1], 
            [1, 2, 7, 2, 1],
            [25, 22, 15, 41, 21]], dtype=np.double)
        query = np.array([
            [1, 5, 6, 1, 1], 
            [25, 2, 15, 41, 21],
            [25, 22, 15, 41, 21], 
            [1, 2, 7, 2, 1]], dtype=np.double)
        dist_params = {'window': 10, 'psi': 0}

        truth = nn_dtw(space, query, dist_params)
        pred = dtw_search.nearest_neighbour_lb_keogh(space, query, None, dist_params=dist_params, use_c=True, use_ucr=True)
        npt.assert_equal(truth, pred)


@pytest.mark.benchmark(group="1NN")
def test_nn_dtw_trace(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)
    dist_params = {'window': 10, 'psi': 0}

    def d():
        return nn_dtw(space, query, dist_params)

    Y = benchmark(d)
    npt.assert_equal(Y, np.array([29, 26]))


@pytest.mark.benchmark(group="1NN")
def test_nn_keogh_trace_python(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)
    dist_params = {'window': 10, 'psi': 0}

    envelopes = dtw_search.lb_keogh_envelope(query, window=dist_params.get('window', None), use_c=False, parallel=True)
    def d():
        return dtw_search.nearest_neighbour_lb_keogh(space, query, envelopes, dist_params=dist_params, use_c=False, parallel=True)

    Y = benchmark(d)
    npt.assert_equal(Y, np.array([29, 26]))


@pytest.mark.benchmark(group="1NN")
def test_nn_keogh_trace_C(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)
    dist_params = {'window': 10, 'psi': 0}

    envelopes = dtw_search.lb_keogh_envelope(query, window=dist_params.get('window', None), use_c=True, parallel=True)
    def d():
        return dtw_search.nearest_neighbour_lb_keogh(space, query, envelopes, dist_params=dist_params, use_c=True, parallel=True)

    Y = benchmark(d)
    npt.assert_equal(Y, np.array([29, 26]))


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
    dist_params = {'window': 10, 'psi': 0}

    def d():
        distmat = dtw.distance_matrix_fast(
                np.concatenate((query, rolling_space), axis=0), 
                block=((0, query.shape[0]), (query.shape[0], rolling_space.shape[0]+query.shape[0])), 
                **dist_params)
        np.fill_diagonal(distmat, np.inf)
        return np.argmin(distmat, axis=0)[:query.shape[0]] - query.shape[0]

    Y = benchmark(d)
    npt.assert_equal(Y, np.array([7976, 7149]))


@pytest.mark.benchmark(group="1NN Subsequence")
def test_nn_ucr_subsequence_trace(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)
    dist_params = {'window': 10, 'psi': 0}

    def d():
        return dtw_search.nearest_neighbour_lb_keogh(space.ravel(), query, None, dist_params=dist_params, use_c=True, use_ucr=True, parallel=True)

    Y = benchmark(d)
    npt.assert_equal(Y, np.array([7976, 7149]))


@pytest.mark.benchmark(group="1NN Subsequence")
def test_nn_keogh_subsequence_trace(benchmark, data_trace):

    query = data_trace[:2,:].astype(np.double)
    space = data_trace[2:,:].astype(np.double)
    dist_params = {'window': 10, 'psi': 0}

    def d():
        return dtw_search.nearest_neighbour_lb_keogh(space.ravel(), query, None, dist_params=dist_params, use_c=True, use_ucr=False, parallel=True)

    Y = benchmark(d)
    npt.assert_equal(Y, np.array([7976, 7149]))


def test_nn_keogh_parallel_bug(data_trace):

    nb_query = 10
    query = data_trace[:nb_query,:].astype(np.double)
    space = data_trace[nb_query:,:].astype(np.double)
    dist_params = {'window': 10, 'psi': 0}

    envelopes = dtw_search.lb_keogh_envelope(query, window=dist_params.get('window', None), use_c=True, parallel=True)
    Y = dtw_search.nearest_neighbour_lb_keogh(space, query, envelopes, dist_params=dist_params, use_c=True, parallel=True)
    truth = nn_dtw(space, query, dist_params)

    npt.assert_equal(Y, truth)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # test_keogh_envelope(True)
    # test_keogh_nn(True)

