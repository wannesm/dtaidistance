import pytest
import os
import random
from pathlib import Path
from dtaidistance import dtw, util_numpy
from dtaidistance import dtw_visualisation as dtwvis


directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_normalize():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
        s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
        r, path = dtw.warp(s1, s2)
        if directory:
            dtwvis.plot_warp(s1, s2, r, path, filename=str(directory / "test_normalize1.png"))
        r_c = np.array([0., 1., 2., 2., 1., 0.5, 0., 0., 2., 1., 0., 0., 0.])
        if directory:
            path = dtw.warping_path(s1, s2, psi=2)
            dtwvis.plot_warping(s1, s2, path, filename=str(directory / "test_normalize2.png"))
        np.testing.assert_almost_equal(r, r_c, decimal=4)


@numpyonly
def test_normalize2():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
        s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
        d1, paths1 = dtw.warping_paths(s1, s2, psi=2)
        d2, paths2 = dtw.warping_paths_fast(s1, s2, psi=2)
        path1 = dtw.best_path(paths1)
        path2 = dtw.best_path(paths2)
        if directory:
            dtwvis.plot_warpingpaths(s1, s2, paths1, path1, filename=directory / "normalize.png")
        np.testing.assert_almost_equal(d1, d2, decimal=4)
        np.testing.assert_almost_equal(paths1, paths2, decimal=4)
        np.testing.assert_almost_equal(path1, path2, decimal=4)


@numpyonly
def test_psi_dtw_1a():
    with util_numpy.test_uses_numpy() as np:
        x = np.arange(0, 20, .5)
        s1 = np.sin(x)
        s2 = np.sin(x - 1)
        # Add noise
        # random.seed(1)
        # for idx in range(len(s2)):
        #     if random.random() < 0.05:
        #         s2[idx] += (random.random() - 0.5) / 2
        d, paths = dtw.warping_paths(s1, s2, psi=2, window=25)
        path = dtw.warping_path(s1, s2, psi=2)
        if directory:
            dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=str(directory / "test_psi_dtw_1a.png"))
        # print(paths[:,:])
        # dtwvis.plot_warping(s1, s2, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_1_1.png"))
        # path = dtw.best_path(paths)
        # dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_1_2.png"))
        np.testing.assert_equal(d, 0.0)


@numpyonly
def test_psi_dtw_1b():
    with util_numpy.test_uses_numpy() as np:
        x = np.arange(0, 20, .5)
        s1 = np.sin(x)
        s2 = np.sin(x - 1)
        d = dtw.distance(s1, s2, psi=2)
        np.testing.assert_equal(d, 0.0)


@numpyonly
def test_psi_dtw_1c():
    with util_numpy.test_uses_numpy() as np:
        x = np.arange(0, 20, .5)
        s1 = np.sin(x)
        s2 = np.sin(x - 1)
        d = dtw.distance_fast(s1, s2, psi=2)
        np.testing.assert_equal(d, 0.0)


@numpyonly
def test_psi_dtw_2a():
    with util_numpy.test_uses_numpy() as np:
        x = np.arange(0, 20, .5)
        s1 = np.sin(x - 1)
        s2 = np.sin(x)
        d, paths = dtw.warping_paths(s1, s2, psi=2, window=3)
        # path = dtw.warping_path(s1, s2, psi=2)
        # dtwvis.plot_warping(s1, s2, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_2_1.png"))
        # path = dtw.best_path(paths)
        # dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_2_2.png"))
        np.testing.assert_equal(d, 0.0)


@numpyonly
def test_psi_dtw_2b():
    with util_numpy.test_uses_numpy() as np:
        x = np.arange(0, 20, .5)
        s1 = np.sin(x - 1)
        s2 = np.sin(x)
        d = dtw.distance(s1, s2, psi=2, window=3)
        np.testing.assert_equal(d, 0.0)


@numpyonly
def test_psi_dtw_2c():
    with util_numpy.test_uses_numpy() as np:
        x = np.arange(0, 20, .5)
        s1 = np.sin(x - 1)
        s2 = np.sin(x)
        d = dtw.distance_fast(s1, s2, psi=2, window=3)
        np.testing.assert_equal(d, 0.0)


@numpyonly
def test_twoleadecg_1(directory=None):
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([1.8896,-0.23712,-0.23712,-0.20134,-0.16556,-0.20134,-0.16556,-0.12978,-0.058224,0.013335,0.031225,0.10278,0.013335,-0.094004,-0.058224,-0.11189,-0.14767,-0.16556,-0.14767,-0.094004,-0.14767,-0.16556,-0.16556,-0.21923,-0.21923,-0.25501,-0.20134,-0.20134,-0.18345,-0.23712,-0.20134,-0.23712,-0.12978,-0.11189,-0.46969,-1.2747,-2.3481,-2.8133,-2.7775,-2.5986,-2.3839,-2.0082,-1.8651,-1.6146,-1.3463,-1.1495,-0.88115,-0.55914,-0.34446,-0.16556,-0.0045548,0.2459,0.53214,0.65737,0.71104,0.74682,0.76471,0.76471,0.80049,0.81838,0.87204,0.88993,0.97938,0.97938,1.0152,1.0867,1.1583,1.1762,1.212,1.2656,1.2656,1.2477,1.2656,1.1762,1.0867,0.99727,0.88993,0.74682,0.63948,0.58581,0.47847,0.38902])
        s2 = np.array([1,0.93163,0.094486,0.094486,0.038006,0.080366,0.080366,0.052126,0.080366,0.12273,0.22157,0.29217,0.41925,0.48985,0.39101,0.39101,0.30629,0.24981,0.19333,0.080366,-0.0043544,-0.018474,-0.089075,-0.11731,-0.14555,-0.17379,-0.21615,-0.27263,-0.20203,-0.315,-0.25851,-0.17379,-0.28675,-0.24439,0.16509,-0.11731,-1.0069,-1.9812,-2.4895,-2.786,-2.9272,-2.4612,-2.0518,-1.8964,-1.8258,-1.7411,-1.6705,-1.2893,-0.99276,-0.65388,-0.37148,-0.30087,-0.046714,0.30629,0.53221,0.65929,0.65929,0.72989,0.74401,0.87109,0.89933,0.95581,0.96993,1.0546,1.1394,1.2523,1.2523,1.2947,1.3088,1.3512,1.2806,1.2806,1.1394,1.097,0.89933,0.72989,0.67341,0.54633,0.37689,0.23569,0.10861,0.080366,-0.074955])
        d, paths = dtw.warping_paths(s1, s2, psi=2, window=5)
        path = dtw.warping_path(s1, s2, psi=2)
        if directory:
            dtwvis.plot_warping(s1, s2, path, filename=str(directory / "warping.png"))
            path = dtw.best_path(paths)
            dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=str(directory / "warpingpaths.png"))


if __name__ == "__main__":
    with util_numpy.test_uses_numpy() as np:
        np.set_printoptions(precision=2, linewidth=120)
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_normalize()
    # test_normalize2()
    test_psi_dtw_1a()
    # test_psi_dtw_1b()
    # test_psi_dtw_1c()
    # test_psi_dtw_2a()
    # test_psi_dtw_2b()
    # test_twoleadecg_1()

