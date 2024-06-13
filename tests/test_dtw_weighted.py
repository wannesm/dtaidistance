from __future__ import print_function
import sys
import logging
import io
import pytest
from pathlib import Path

from dtaidistance import dtw_weighted as dtww
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw, util_numpy
from dtaidistance.util import prepare_directory


logger = logging.getLogger("be.kuleuven.dtai.distance")
directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


def plot_series(s, l, idx=None):
    global directory
    if directory is None:
        directory = prepare_directory()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib not installed')
        return
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig1, ax1 = plt.subplots(nrows=len(s), ncols=1)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    for i, si in enumerate(s):
        if i == idx:
            color = colors[0]
        else:
            color = colors[int(1 + l[i])]
        ax1[i].plot(si, color=color)
        ax2.plot(si, color=color)
    fig1.savefig(str(directory / "series1.png"))
    fig2.savefig(str(directory / "series2.png"))


def plot_margins(serie, weights, clfs, importances=None):
    global directory
    if directory is None:
        directory = prepare_directory()
    try:
        from sklearn import tree
    except ImportError:
        return
    feature_names = ["f{} ({}, {})".format(i // 2, i, '-' if (i % 2) == 0 else '+') for i in range(2 * len(serie))]
    out_str = io.StringIO()
    for clf in clfs:
        tree.export_graphviz(clf, out_file=out_str, feature_names=feature_names)
        print("\n\n", file=out_str)
    with open(str(directory / "tree.dot"), "w") as ofile:
        print(out_str.getvalue(), file=ofile)
    dtww.plot_margins(serie, weights, filename=str(directory / "margins.png"), importances=importances)


@pytest.mark.skip("Ignore weighted")
@numpyonly
def test_distance1():
    with util_numpy.test_uses_numpy() as np:
        directory = prepare_directory()

        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
        s2 = np.array([0., 1, 2, 3, 1, 10, 1, 0, 2, 1, 0, 0, 0])
        d, paths = dtw.warping_paths(s1, s2)
        # print(d, "\n", paths)
        if not dtwvis.test_without_visualization():
            dtwvis.plot_warpingpaths(s1, s2, paths, filename=directory / "temp1.png")

        weights = np.full((len(s1), 8), np.inf)
        weights[:, 2:4] = 0.0
        weights[4:7, 2:4] = 10.0
        weights[:, 4:6] = 0.0
        weights[4:7, 4:6] = 10.0
        d, paths = dtww.warping_paths(s1, s2, weights)
        # print(d, "\n", paths)
        if not dtwvis.test_without_visualization():
            dtwvis.plot_warpingpaths(s1, s2, paths, filename=directory / "temp2.png")


@pytest.mark.skip("Ignore weighted")
@numpyonly
def test_distance2():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
            [0., 0, 1, 2, 1, 0, 1.3, 0, 0],
            [0., 0, 1, 2, 1, 0, 1,   0, 0],
            [0., 1, 2, 0, 0, 0, 0,   0, 0],
            [0., 1, 2, 0, 0, 0, 0,   0, 0],
            [1., 2, 0, 0, 0, 0, 0,   1, 1],
            [1., 2, 0, 0, 0, 0, 0,   1, 1],
            [1., 2, 0, 0, 1, 0, 0,   1, 1]])
        l = np.array([1, 1, 1, 1, 0, 0, 0])

        if directory:
            if not dtwvis.test_without_visualization():
                plot_series(s, l)
            savefig = str(directory / "dts.dot")
        else:
            savefig = None

        prototypeidx = 0
        ml_values, cl_values, clfs, importances = \
            dtww.series_to_dt(s, l, prototypeidx, max_clfs=50, savefig=savefig)
        # logger.debug(f"ml_values = {dict(ml_values)}")
        # logger.debug(f"cl_values = {dict(cl_values)}")
        weights = dtww.compute_weights_from_mlclvalues(s[prototypeidx], ml_values, cl_values, only_max=False, strict_cl=True)

        if not dtwvis.test_without_visualization():
            if directory:
                plot_margins(s[prototypeidx], weights, clfs)


@pytest.mark.skip("Ignore weighted")
@numpyonly
def test_distance3():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
            [0., 0, 1, 2, 1, 0, 1.3, 0, 0],
            [0., 1, 2, 0, 0, 0, 0, 0, 0]
        ])
        w = np.array([[np.inf, np.inf, 0.,  0., 0.,  0.,   np.inf, np.inf],
                      [np.inf, np.inf, 1.1, 1., 0.,  0.,   np.inf, np.inf],
                      [np.inf, np.inf, 1.1, 1., 0.,  0.,   np.inf, np.inf],
                      [np.inf, np.inf, 0.,  0., 2.,  2.2,  np.inf, np.inf],
                      [np.inf, np.inf, 0.,  0., 1.,  1.1,  np.inf, np.inf],
                      [np.inf, np.inf, 0.,  0., 0.,  0.,   np.inf, np.inf],
                      [np.inf, np.inf, 0.,  0., 1.3, 1.43, np.inf, np.inf],
                      [np.inf, np.inf, 0.,  0., 0.,  0.,   np.inf, np.inf],
                      [np.inf, np.inf, 0.,  0., 0.,  0.,   np.inf, np.inf]])

        d, paths = dtww.warping_paths(s[0], s[1], w, window=0)
        path = dtw.best_path(paths)
        if not dtwvis.test_without_visualization():
            if directory:
                wp_fn = directory / "warping_paths.png"
                dtwvis.plot_warpingpaths(s[0], s[1], paths, path, filename=wp_fn)


@pytest.mark.skip("Ignore weighted")
@numpyonly
def test_distance4():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
            [0., 0, 1,    2,    1,   0,   1.3, 0, 0],  # 0
            [0., 1, 2,    0,    0,   0,   0,   0, 0],  # 1
            [1., 2, 0,    0,    0,   0,   0,   1, 1],  # 2
            [0., 0, 1,    2,    1,   0,   1,   0, 0],  # 3
            [0., 1, 2,    0,    0,   0,   0,   0, 0],  # 4
            [1., 2, 0,    0,    0,   0,   0,   1, 1],  # 5
            [1., 2, 0,    0,    1,   0,   0,   1, 1],  # 6
            [1., 2, 0.05, 0.01, 0.9, 0,   0,   1, 1]]) # 7
        l = np.array([1, 0, 0, 1, 0, 0, 0, 0])

        if directory:
            if not dtwvis.test_without_visualization():
                plot_series(s, l)
            savefig = str(directory / "dts.dot")
        else:
            savefig = None

        prototypeidx = 0
        ml_values, cl_values, clf, importances = \
            dtww.series_to_dt(s, l, prototypeidx, window=2, min_ig=0.1, savefig=savefig)
        # logger.debug(f"ml_values = {dict(ml_values)}")
        # logger.debug(f"cl_values = {dict(cl_values)}")
        weights = dtww.compute_weights_from_mlclvalues(s[prototypeidx], ml_values, cl_values,
                                                       only_max=False, strict_cl=True)
        if directory:
            if not dtwvis.test_without_visualization():
                plot_margins(s[prototypeidx], weights, clf)


@pytest.mark.skip("Ignore weighted")
@numpyonly
def test_distance5():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
            [0., 0, 0, 2,  0, -2, 0, 0,  0, 0, 0, 0,  0, 0, 0],  # 0
            [0., 0, 2, 0, -2,  0, 2, 0, -2, 0, 2, 0, -2, 0, 0],  # 1
            [0., 0, 2, 0,  0,  0, 2, 0,  0, 0, 2, 0,  0, 0, 0]   # 2
        ])
        l = np.array([1, 1, 0])

        if directory:
            if not dtwvis.test_without_visualization():
                plot_series(s, l)

        prototypeidx = 0
        ml_values, cl_values, clf, importances = dtww.series_to_dt(s, l, prototypeidx, window=4)
        # logger.debug(f"ml_values = {dict(ml_values)}")
        # logger.debug(f"cl_values = {dict(cl_values)}")
        weights = dtww.compute_weights_from_mlclvalues(s[prototypeidx], ml_values, cl_values,
                                                       only_max=False, strict_cl=True)
        if directory:
            if not dtwvis.test_without_visualization():
                plot_margins(s[prototypeidx], weights, clf)


@pytest.mark.skip("Takes too long")
def test_distance6():
    with util_numpy.test_uses_numpy() as np:
        s = np.loadtxt(Path(__file__).parent / "rsrc" / "series_0.csv", delimiter=',')
        l = np.loadtxt(Path(__file__).parent / "rsrc" / "labels_0.csv", delimiter=',')

        if directory:
            if not dtwvis.test_without_visualization():
                plot_series(s, l)
            savefig = str(directory / "dts.dot")
        else:
            savefig = None

        prototypeidx = 3
        labels = np.zeros(l.shape)
        labels[l == l[prototypeidx]] = 1
        ml_values, cl_values, clf, importances = \
            dtww.series_to_dt(s, labels, prototypeidx, window=0, min_ig=0.1, savefig=savefig)
        # logger.debug(f"ml_values = {dict(ml_values)}")
        # logger.debug(f"cl_values = {dict(cl_values)}")
        weights = dtww.compute_weights_from_mlclvalues(s[prototypeidx], ml_values, cl_values,
                                                       only_max=False, strict_cl=True)
        if directory:
            if not dtwvis.test_without_visualization():
                plot_margins(s[prototypeidx], weights, clf, prototypeidx)


@pytest.mark.skip("Ignore weighted")
@numpyonly
def test_distance7():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
            [0.0, 0.3, 0.5, 0.8, 1.0, 0.1, 0.0, 0.1],
            [0.0, 0.2, 0.3, 0.7, 1.1, 0.0, 0.1, 0.0],
            [0.1, 0.0, 1.0, 1.0, 1.0, 0.9, 0.0, 0.0],
            [0.0, 0.0, 1.1, 0.9, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.1, 1.1, 1.0, 0.9, 0.9, 0.0, 0.0],
            [0.0, 0.1, 1.0, 1.1, 0.9, 1.0, 0.0, 0.1],
            [0.0, 0.1, 0.4, 0.3, 0.2, 0.3, 0.0, 0.0],
            [0.1, 0.0, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1]])
        l = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        prototypeidx = 0

        if directory:
            if not dtwvis.test_without_visualization():
                plot_series(s, l, prototypeidx)
            savefig = str(directory / "dts.dot")
        else:
            savefig = None
        ml_values, cl_values, clf, imp = dtww.series_to_dt(s, l, prototypeidx, window=0, min_ig=0.01,
                                                           savefig=savefig,
                                                           warping_paths_fnc=dtww.warping_paths)
        # logger.debug(f"ml_values = {dict(ml_values)}")
        # logger.debug(f"cl_values = {dict(cl_values)}")
        weights = dtww.compute_weights_from_mlclvalues(s[prototypeidx], ml_values, cl_values,
                                                       only_max=False, strict_cl=True)
        if directory:
            if not dtwvis.test_without_visualization():
                plot_margins(s[prototypeidx], weights, clf, imp)


if __name__ == "__main__":
    # Print options
    # np.set_printoptions(precision=2)
    # Logger options
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = 0
    # Output path
    directory = Path(__file__).resolve().parent.parent / "tests" / "output"

    # Functions
    # test_distance1()
    # test_distance2()
    # test_distance3()
    # test_distance4()
    # test_distance5()
    # test_distance6()
    test_distance7()
