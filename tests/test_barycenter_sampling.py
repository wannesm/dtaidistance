import pytest

from dtaidistance import dtw_barycenter, util_numpy


numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
@pytest.mark.parametrize("use_c", [False, True])
@pytest.mark.parametrize("nb_initial_samples", [3, 10])
def test_initial_samples_respect_mask(use_c, nb_initial_samples):
    with util_numpy.test_uses_numpy() as np:
        series = np.array([[100.], [0.], [2.], [3.], [200.]])
        mask = np.array([False, True, True, True, False])
        with np.errstate(divide='raise', invalid='raise'):
            result = dtw_barycenter.get_good_c(
                series, mask, nb_initial_samples, use_c=use_c)
        np.testing.assert_array_equal(result, series[2])


@numpyonly
@pytest.mark.parametrize("use_c", [False, True])
def test_initial_samples_return_sampled_medoid(monkeypatch, use_c):
    with util_numpy.test_uses_numpy() as np:
        series = np.array([[100.], [100.], [0.], [2.], [3.]])
        mask = np.ones(len(series), dtype=bool)
        draws = iter([1., 1., 0., 0., 0.])
        monkeypatch.setattr(dtw_barycenter.random, 'random', lambda: next(draws))
        result = dtw_barycenter.get_good_c(series, mask, 3, use_c=use_c)
        np.testing.assert_array_equal(result, series[3])
