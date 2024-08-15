import pytest
import os
import random
import itertools
import multiprocessing as mp
import functools
from pathlib import Path

from dtaidistance import dtw, util_numpy
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance.exceptions import MatplotlibException


directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_normalize():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
        s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
        r, path1 = dtw.warp(s1, s2)
        path2 = dtw.warping_path(s1, s2, psi=2)
        if not dtwvis.test_without_visualization():
            if directory:
                dtwvis.plot_warp(s1, s2, r, path1, filename=str(directory / "test_normalize1.png"))
                dtwvis.plot_warping(s1, s2, path2, filename=str(directory / "test_normalize2.png"))
        r_c = np.array([0., 1., 2., 2., 1., 0.5, 0., 0., 2., 1., 0., 0., 0.])
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
        if not dtwvis.test_without_visualization():
            if directory:
                dtwvis.plot_warpingpaths(s1, s2, paths1, path1, filename=directory / "normalize.png")
        np.testing.assert_almost_equal(d1, d2, decimal=4)
        np.testing.assert_almost_equal(paths1, paths2, decimal=4)
        np.testing.assert_almost_equal(path1, path2, decimal=4)


@numpyonly
def test_normalize2_prob():
    psi = 0
    if dtw.dtw_cc is not None:
        dtw.dtw_cc.srand(random.randint(1, 100000))
    else:
        print("WARNING: dtw_cc not found")
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
        s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
        d1, paths1 = dtw.warping_paths(s1, s2, psi=psi)
        d2, paths2 = dtw.warping_paths_fast(s1, s2, psi=psi)
        # print(np.power(paths1,2))
        path1 = dtw.best_path(paths1)
        path2 = dtw.best_path(paths2)
        prob_paths = []
        for i in range(30):
            prob_paths.append(dtw.warping_path_prob(s1, s2, d1/len(s1), psi=psi))
        if not dtwvis.test_without_visualization():
            if directory:
                fig, ax = dtwvis.plot_warpingpaths(s1, s2, paths1, path1)
                for p in prob_paths:
                    py, px = zip(*p)
                    py = [pyi + (random.random() - 0.5) / 5 for pyi in py]
                    px = [pxi + (random.random() - 0.5) / 5 for pxi in px]
                    ax[3].plot(px, py, ".-", color="yellow", alpha=0.25)
                fig.savefig(directory / "normalize2_prob.png")
        np.testing.assert_almost_equal(d1, d2, decimal=4)
        np.testing.assert_almost_equal(paths1, paths2, decimal=4)
        np.testing.assert_almost_equal(path1, path2, decimal=4)

@numpyonly
def test_warping_path1():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 0, 0])
        s2 = np.array([0., 1, 2, 0, 0, 0, 0, 0, 0, 0, 0])
        path1, d1 = dtw.warping_path(s1, s2, include_distance=True)
        path2, d2 = dtw.warping_path_fast(s1, s2, include_distance=True)
        path3 = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
        assert len(path1) == len(path3)
        assert len(path2) == len(path3)
        assert d1 == pytest.approx(d2), "{} != {}".format(d1, d2)
        assert all(ai1 == bi1 and ai2 == bi2 for ((ai1, ai2), (bi1, bi2)) in zip(path1, path3))
        assert all(ai1 == bi1 and ai2 == bi2 for ((ai1, ai2), (bi1, bi2)) in zip(path2, path3))


@numpyonly
def test_warping_path2():
    with util_numpy.test_uses_numpy() as np:
        x = np.array([1., 2, 2, 4, 5, 5])
        y = np.array([1., 2, 2, 4, 4, 4, 5])
        # x = np.random. randn (5)
        # y = np. random. randn (6)
        dist = dtw.distance(x, y, window=2)
        dist_fast = dtw.distance_fast(x, y, window=2)
        path, dist_a = dtw.warping_path(x, y, include_distance=True, window=2)
        path_fast, dist_fast_a = dtw.warping_path_fast(x, y, include_distance=True, window=2)
        d_wps, wps = dtw.warping_paths(x, y, window=2)
        d_wps_fast, wps_fast = dtw.warping_paths_fast(x, y, window=2)
        assert d_wps == pytest.approx(d_wps_fast)
        assert dist == pytest.approx(dist_fast)
        assert dist == pytest.approx(dist_a)
        assert dist == pytest.approx(dist_fast_a)
        assert str(path) == str(path_fast)


def compute_path(s, dtw_settings, idx):
    ri, ci = idx
    if ci <= ri:
        return None
    return dtw.warping_path(s[ci], s[ri], **dtw_settings)


def compute_path_len(s, dtw_settings, idx):
    ri, ci = idx
    if ci <= ri:
        return 0, 0
    path = dtw.warping_path(s[ci], s[ri], **dtw_settings)
    return len(path), 1


@numpyonly
def test_warping_path_matrix():
    with util_numpy.test_uses_numpy() as np:
        s = np.array(
            [[0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1]])

        # compute and store all paths first
        with mp.Pool() as pool:
            dtw_settings = {}
            paths = pool.map(functools.partial(compute_path, s, dtw_settings),
                             itertools.product(range(len(s)), repeat=2))

        # compute paths one by one and immediately reduce to some target
        avg_len, avg_len_cnt = 0, 0
        for ri, ci in itertools.product(range(len(s)), repeat=2):
            if ci <= ri:
                continue
            path = dtw.warping_path(s[ci], s[ri], **dtw_settings)
            avg_len += len(path)
            avg_len_cnt += 1
        avg_len /= avg_len_cnt

        # reduce operation in parallel (in this case simple because the operator is commutative and associative)
        with mp.Pool() as pool:
            dtw_settings = {}
            len_cnt = pool.map(functools.partial(compute_path_len, s, dtw_settings),
                               itertools.product(range(len(s)), repeat=2))
            lens, cnts = zip(*len_cnt)
            avg_len2 = sum(lens) / sum(cnts)

        assert avg_len == pytest.approx(avg_len2)


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
        path_fast = dtw.warping_path_fast(s1, s2, psi=2)
        if not dtwvis.test_without_visualization():
            if directory:
                dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=str(directory / "test_psi_dtw_1a.png"))
            # print(paths[:,:])
            # dtwvis.plot_warping(s1, s2, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_1_1.png"))
            # path = dtw.best_path(paths)
            # dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_1_2.png"))
        np.testing.assert_equal(d, 0.0)
        assert str(path) == str(path_fast)


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
def test_psi_dtw_1d():
    with util_numpy.test_uses_numpy() as np:
        x = np.arange(0, 20, .5)
        s1 = np.sin(x)
        s2 = np.sin(x - 1)

        random.seed(1)
        for idx in range(len(s2)):
            if random.random() < 0.05:
                s2[idx] += (random.random() - 0.5) / 2

        # print(f's1 = [' + ','.join(f'{vv:.2f}' for vv in s1) + ']')
        # print(f's2 = [' + ','.join(f'{vv:.2f}' for vv in s2) + ']')

        # print('distance_fast')
        d1 = dtw.distance_fast(s1, s2, psi=2)
        # print(f'{d1=}')
        # print('warping_paths')
        d2, paths = dtw.warping_paths(s1, s2, window=25, psi=2)
        # print(f'{d2=}')
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            print(paths)
        # print('warping_paths fast')
        d3, paths = dtw.warping_paths_fast(s1, s2, window=25, psi=2)
        # print(f'{d3=}')
        # print(paths)
        # print('best_path')
        best_path = dtw.best_path(paths)

        if not dtwvis.test_without_visualization():
            if directory:
                dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename=directory / "test_psi_dtw_1d.png")

        np.testing.assert_almost_equal(d1, d2)
        np.testing.assert_almost_equal(d1, d3)


@numpyonly
def test_psi_dtw_2a():
    with util_numpy.test_uses_numpy() as np:
        x = np.arange(0, 20, .5)
        s1 = np.sin(x - 1)
        s2 = np.sin(x)
        d, paths = dtw.warping_paths(s1, s2, psi=2, window=3)
        # try:
        path = dtw.warping_path(s1, s2, psi=2)
        path_fast = dtw.warping_path_fast(s1, s2, psi=2)
        # dtwvis.plot_warping(s1, s2, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_2_1.png"))
        # path = dtw.best_path(paths)
        # dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=os.path.expanduser("~/Desktop/test_psi_dtw_2_2.png"))
        # except MatplotlibException:
        # pass
        np.testing.assert_equal(d, 0.0)
        assert str(path) == str(path_fast)


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
def test_twoleadecg_1():
    """Example from http://www.timeseriesclassification.com/description.php?Dataset=TwoLeadECG"""
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([1.8896,-0.23712,-0.23712,-0.20134,-0.16556,-0.20134,-0.16556,-0.12978,-0.058224,0.013335,0.031225,0.10278,0.013335,-0.094004,-0.058224,-0.11189,-0.14767,-0.16556,-0.14767,-0.094004,-0.14767,-0.16556,-0.16556,-0.21923,-0.21923,-0.25501,-0.20134,-0.20134,-0.18345,-0.23712,-0.20134,-0.23712,-0.12978,-0.11189,-0.46969,-1.2747,-2.3481,-2.8133,-2.7775,-2.5986,-2.3839,-2.0082,-1.8651,-1.6146,-1.3463,-1.1495,-0.88115,-0.55914,-0.34446,-0.16556,-0.0045548,0.2459,0.53214,0.65737,0.71104,0.74682,0.76471,0.76471,0.80049,0.81838,0.87204,0.88993,0.97938,0.97938,1.0152,1.0867,1.1583,1.1762,1.212,1.2656,1.2656,1.2477,1.2656,1.1762,1.0867,0.99727,0.88993,0.74682,0.63948,0.58581,0.47847,0.38902])
        s2 = np.array([1,0.93163,0.094486,0.094486,0.038006,0.080366,0.080366,0.052126,0.080366,0.12273,0.22157,0.29217,0.41925,0.48985,0.39101,0.39101,0.30629,0.24981,0.19333,0.080366,-0.0043544,-0.018474,-0.089075,-0.11731,-0.14555,-0.17379,-0.21615,-0.27263,-0.20203,-0.315,-0.25851,-0.17379,-0.28675,-0.24439,0.16509,-0.11731,-1.0069,-1.9812,-2.4895,-2.786,-2.9272,-2.4612,-2.0518,-1.8964,-1.8258,-1.7411,-1.6705,-1.2893,-0.99276,-0.65388,-0.37148,-0.30087,-0.046714,0.30629,0.53221,0.65929,0.65929,0.72989,0.74401,0.87109,0.89933,0.95581,0.96993,1.0546,1.1394,1.2523,1.2523,1.2947,1.3088,1.3512,1.2806,1.2806,1.1394,1.097,0.89933,0.72989,0.67341,0.54633,0.37689,0.23569,0.10861,0.080366,-0.074955])
        kwargs = {'psi': 2, 'window': 5}
        d, paths = dtw.warping_paths(s1, s2, **kwargs)
        path = dtw.warping_path(s1, s2, **kwargs)
        path_fast = dtw.warping_path_fast(s1, s2, **kwargs)
        if not dtwvis.test_without_visualization():
            if directory:
                import matplotlib.pyplot as plt
                fig, axs = dtwvis.plot_warping(s1, s2, path)  # type: plt.Figure, plt.axes.Axes
                fig.set_size_inches(12, 10)
                fig.set_dpi(100)
                fig.savefig(str(directory / "warping.png"))
                plt.close(fig)
                path = dtw.best_path(paths)
                dtwvis.plot_warpingpaths(s1, s2, paths, path, filename=str(directory / "warpingpaths.png"))

        d, wps = dtw.warping_paths(s1, s2, **kwargs)
        wps_best_path = dtw.best_path(wps)
        d_fast, wps_fast = dtw.warping_paths_fast(s1, s2, **kwargs)
        wps_fast_best_path = dtw.best_path(wps_fast)
        d_fast, wps_fast_c = dtw.warping_paths_fast(s1, s2, compact=True, **kwargs)
        wps_fast_c_best_path = dtw.dtw_cc.best_path_compact(wps_fast_c, len(s1), len(s2), **kwargs)
        path4, d4 = dtw.warping_path(s1, s2, include_distance=True, **kwargs)
        path5, d5 = dtw.warping_path_fast(s1, s2, include_distance=True, **kwargs)

        assert str(wps_best_path) == str(wps_fast_best_path)
        assert str(wps_best_path) == str(wps_fast_c_best_path)
        assert str(wps_best_path) == str(path4)
        assert str(wps_best_path) == str(path5)
        np.testing.assert_allclose(wps, wps_fast)
        assert str(path) == str(path_fast)
        assert d == pytest.approx(d_fast)
        assert d == pytest.approx(d4)
        assert d == pytest.approx(d5)

@numpyonly
def test_subsequence():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([1., 2, 0])
        s2 = np.array([1., 0, 1, 2, 1, 0, 1, 0, 0, 0, 0])
        penalty = 0.1
        psi = [0, 0, len(s2), len(s2)]
        d1, paths1 = dtw.warping_paths(s1, s2, penalty=penalty, psi=psi)
        d2, paths2 = dtw.warping_paths_fast(s1, s2, penalty=penalty, psi=psi)
        path1 = dtw.best_path(paths1)
        path2 = dtw.best_path(paths2)
        if not dtwvis.test_without_visualization():
            if directory:
                dtwvis.plot_warpingpaths(s1, s2, paths1, path1, filename=directory / "subseq.png")
        np.testing.assert_almost_equal(d1, d2, decimal=4)
        np.testing.assert_almost_equal(paths1, paths2, decimal=4)
        np.testing.assert_almost_equal(path1, path2, decimal=4)
        np.testing.assert_almost_equal(paths1[3:4, 0:12][0],
            [np.inf,1.421,1.005,1.421,2.002,1.000,-1,-1,-1,-1,-1,-1], decimal=3)


if __name__ == "__main__":
    with util_numpy.test_uses_numpy() as np:
        np.set_printoptions(precision=2, linewidth=120)
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print("Saving files to {}".format(directory))
    # test_normalize()
    test_normalize2()
    # test_normalize2_prob()
    # test_warping_path1()
    # test_warping_path2()
    # test_psi_dtw_1a()
    # test_psi_dtw_1b()
    # test_psi_dtw_1c()
    # test_psi_dtw_1d()
    # test_psi_dtw_2a()
    # test_psi_dtw_2b()
    # test_twoleadecg_1()
    # test_subsequence()
