import logging
import pytest
import numpy as np

from dtaidistance import dtw_search, util_numpy


logger = logging.getLogger("be.kuleuven.dtai.distance")
directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


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
def test_keogh_distance_equal_lenghts_iterator(use_c, parallel):
    with util_numpy.test_uses_numpy() as np:
      ts1 = np.array([[1, 2, 3, 2, 1]], dtype=np.double)
      ts2 = np.array([[0, 0, 0, 0, 0]], dtype=np.double)
      d = dtw_search.lb_keogh(ts2, ts1, use_c=use_c, parallel=parallel)
      assert d == [[5]]


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
@pytest.mark.parametrize("use_c", [True, False])
def test_keogh_nn(use_c):
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

        Y = dtw_search.nearest_neighbour_lb_keogh(X, X2, None, distParams={'window': 2, 'psi': 0}, use_c=use_c)

        assert np.array_equal(Y, np.array([0,2,2,1], dtype=np.int))


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    test_keogh_envelope(True)
    test_keogh_nn(True)

