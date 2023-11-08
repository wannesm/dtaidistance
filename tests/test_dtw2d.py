import logging
import sys
import pytest

from dtaidistance import dtw, util_numpy, dtw_ndim, dtw_ndim_visualisation as dtwndimvis
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance.exceptions import MatplotlibException


numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")
logger = logging.getLogger("be.kuleuven.dtai.distance")


@numpyonly
def test_distance1_a():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1],  [0, 0]], dtype=np.double)
        s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
        d1 = dtw_ndim.distance(s1, s2)
        d1p, paths = dtw_ndim.warping_paths(s1, s2)
        # print(d1, d1p)
        assert d1 == pytest.approx(d1p)


@numpyonly
def test_distance1_b():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1],  [0, 0]], dtype=np.double)
        s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
        d1 = dtw_ndim.distance_fast(s1, s2)
        # print(d1)


@numpyonly
def test_distance2():
    test_A = [[2.65598039, 0.93622549, -0.04169118],
              [2.02625, 0.965625, -0.050625],
              [0.41973039, 0.85968137, 0.38970588],
              [0.3669697, -0.03221591, 0.09734848],
              [0.68905971, -0.65101381, -0.70944742],
              [0.14142316, -0.81769481, -0.44556277],
              [0.2569697, -0.6330303, -0.35503788],
              [-0.39339286, 0.02303571, -0.48428571],
              [-0.8275, -0.02125, -0.0325],
              [-0.65293269, -0.29504808, 0.30557692]]

    test_B = [[-1.54647436e+00, -3.76602564e-01, -8.58974359e-01],
              [-1.14283907e+00, -8.50961538e-01, -5.42974022e-01],
              [-4.86715587e-01, -8.62221660e-01, -6.32211538e-01],
              [3.54672740e-02, -4.37500000e-01, -4.41801619e-01],
              [7.28618421e-01, -4.93421053e-03, -8.90625000e-01],
              [1.03525641e+00, 1.25000000e-01, -8.50961538e-01],
              [5.24539474e-01, 1.07828947e-01, -3.99375000e-01],
              [5.04464286e-01, 3.76275510e-01, -6.74744898e-01],
              [1.20897959e+00, 1.10793367e+00, -1.45681122e+00],
              [8.70535714e-01, 8.73724490e-01, -1.01275510e+00]]

    with util_numpy.test_uses_numpy() as np:
        test_A = np.array(test_A)
        test_B = np.array(test_B)

        d1 = dtw_ndim.distance(test_A, test_B, use_c=False)
        d2 = dtw_ndim.distance(test_A, test_B, use_c=True)

        d3, paths = dtw_ndim.warping_paths(test_A, test_B)
        d4, paths = dtw_ndim.warping_paths(test_A, test_B, use_c=True)

        assert d1 == pytest.approx(d2)
        assert d1 == pytest.approx(d3)
        assert d1 == pytest.approx(d4)


@numpyonly
def test_distance3():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([[0., 0.], [1., 2.], [1., 0.], [1., 0.]])
        s2 = np.array([[0., 1.], [2., 0.], [0., 0.], [0., 0.]])

        d1 = dtw_ndim.distance(s1, s2)
        print(d1)

        d2,  paths2 = dtw_ndim.warping_paths(s1, s2)
        print(d2)
        print(paths2)

        path3 = dtw_ndim.warping_path(s1, s2)
        print(path3)

@numpyonly
def test_visualisation_a():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1], [0, 0]], dtype=np.double)
        s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
        d1p, paths = dtw_ndim.warping_paths(s1, s2)
        path = dtw.best_path(paths)
        if not dtwvis.test_without_visualization():
            fig, ax = dtwndimvis.plot_warping(s1, s2, path)
            fig.show()


@numpyonly
def test_visualisation_b():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1], [0, 0]], dtype=np.double)
        s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
        d1p, paths = dtw_ndim.warping_paths(s1, s2)
        path = dtw.best_path(paths)
        if not dtwvis.test_without_visualization():
            fig, ax = dtwndimvis.plot_warpingpaths(s2, s1, paths, path=path)
            fig.show()

@numpyonly
def test_distances1_python():
    with util_numpy.test_uses_numpy() as np:
        s = np.array(
            [[[0., 0], [1, 2], [1, 0], [1, 0]],
             [[0., 1], [2, 0], [0, 0], [0, 0]],
             [[1., 2], [0, 0], [0, 0], [0, 1]],
             [[0., 0], [1, 2], [1, 0], [1, 0]],
             [[0., 1], [2, 0], [0, 0], [0, 0]],
             [[1., 2], [0, 0], [0, 0], [0, 1]]])
        m = dtw_ndim.distance_matrix(s, 2, compact=True)
        assert m[0] == pytest.approx(2.44948974, abs=1e-3)
        assert m[1] == pytest.approx(3.0000)
        assert m[2] == pytest.approx(0.0000)
        assert m[3] == pytest.approx(2.4495, abs=1e-3)
        assert m[4] == pytest.approx(3.0000)


@numpyonly
def test_distances1_fast():
    with util_numpy.test_uses_numpy() as np:
        s = np.array(
            [[[0., 0], [1, 2], [1, 0], [1, 0]],
             [[0., 1], [2, 0], [0, 0], [0, 0]],
             [[1., 2], [0, 0], [0, 0], [0, 1]],
             [[0., 0], [1, 2], [1, 0], [1, 0]],
             [[0., 1], [2, 0], [0, 0], [0, 0]],
             [[1., 2], [0, 0], [0, 0], [0, 1]]])
        m = dtw_ndim.distance_matrix_fast(s, 2, compact=True, parallel=False)
        # print(m)
        assert m[0] == pytest.approx(2.44948974, abs=1e-3)
        assert m[1] == pytest.approx(3.0000)
        assert m[2] == pytest.approx(0.0000)
        assert m[3] == pytest.approx(2.4495, abs=1e-3)
        assert m[4] == pytest.approx(3.0000)


@numpyonly
def test_distances1_fast_parallel():
    with util_numpy.test_uses_numpy() as np:
        s = np.array(
            [[[0., 0], [1, 2], [1, 0], [1, 0]],
             [[0., 1], [2, 0], [0, 0], [0, 0]],
             [[1., 2], [0, 0], [0, 0], [0, 1]],
             [[0., 0], [1, 2], [1, 0], [1, 0]],
             [[0., 1], [2, 0], [0, 0], [0, 0]],
             [[1., 2], [0, 0], [0, 0], [0, 1]]])
        m = dtw_ndim.distance_matrix_fast(s, 2, compact=True, parallel=True)
        # print(m)
        assert m[0] == pytest.approx(2.44948974, abs=1e-3)
        assert m[1] == pytest.approx(3.0000)
        assert m[2] == pytest.approx(0.0000)
        assert m[3] == pytest.approx(2.4495, abs=1e-3)
        assert m[4] == pytest.approx(3.0000)


@numpyonly
def test_distances2_fast():
    with util_numpy.test_uses_numpy() as np:
        s = [[[0., 0], [1, 2], [1, 0], [1, 0]],
             [[0., 1], [2, 0], [0, 0], [0, 0]],
             [[1., 2], [0, 0], [0, 0], [0, 1]],
             [[0., 0], [1, 2], [1, 0], [1, 0]],
             [[0., 1], [2, 0], [0, 0], [0, 0]],
             [[1., 2], [0, 0], [0, 0], [0, 1]]]
        s = [np.array(a) for a in s]
        m = dtw_ndim.distance_matrix_fast(s, 2, compact=True, parallel=False)
        # print(m)
        assert m[0] == pytest.approx(2.44948974, abs=1e-3)
        assert m[1] == pytest.approx(3.0000)
        assert m[2] == pytest.approx(0.0000)
        assert m[3] == pytest.approx(2.4495, abs=1e-3)
        assert m[4] == pytest.approx(3.0000)


@numpyonly
def test_distances2_fast_parallel():
    with util_numpy.test_uses_numpy() as np:
        s = [np.array([[0., 0], [1, 2], [1, 0], [1, 0]]),
             np.array([[0., 1], [2, 0], [0, 0], [0, 0]]),
             np.array([[1., 2], [0, 0], [0, 0], [0, 1]]),
             np.array([[0., 0], [1, 2], [1, 0], [1, 0]]),
             np.array([[0., 1], [2, 0], [0, 0], [0, 0]]),
             np.array([[1., 2], [0, 0], [0, 0], [0, 1]])]
        m = dtw_ndim.distance_matrix_fast(s, 2, compact=True, parallel=True)
        # print(m)
        assert m[0] == pytest.approx(2.44948974, abs=1e-3)
        assert m[1] == pytest.approx(3.0000)
        assert m[2] == pytest.approx(0.0000)
        assert m[3] == pytest.approx(2.4495, abs=1e-3)
        assert m[4] == pytest.approx(3.0000)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # test_distance1_a()
    # test_distance1_b()
    # test_distance2()
    test_distance3()
    # test_visualisation_a()
    # test_visualisation_b()
    # test_distances2_fast()
    # test_distances2_fast_parallel()
