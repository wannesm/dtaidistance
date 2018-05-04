from __future__ import print_function
import os
import sys
import logging
import io
import pytest
import numpy as np
try:
    from pathlib import Path
except ImportError:
    try:
        from pathlib2 import Path  # For Python2
    except ImportError:
        raise ImportError("No pathlib or pathlib2 found")
from dtaidistance import dtw_weighted as dtww
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw
from dtaidistance.util import prepare_directory


logger = logging.getLogger("be.kuleuven.dtai.distance")


def plot_series(s, l):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=len(s), ncols=1)
    for i, si in enumerate(s):
        ax[i].plot(si, color="blue" if l[i] == 1 else "red")
    plt.savefig(str(directory / "series.png"))


def plot_margins(serie, weights, clfs):
    from sklearn import tree
    feature_names = ["f{} ({})".format(i // 2, i) for i in range(2 * len(serie) + 1)]
    out_str = io.StringIO()
    for clf in clfs:
        tree.export_graphviz(clf, out_file=out_str, feature_names=feature_names)
        print("\n\n", file=out_str)
    with open(str(directory / "tree.dot"), "w") as ofile:
        print(out_str.getvalue(), file=ofile)
    dtww.plot_margins(serie, weights, filename=str(directory / "margins.png"))


def test_distance1(directory=None):
    directory = prepare_directory(directory)

    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
    s2 = np.array([0., 1, 2, 3, 1, 10, 1, 0, 2, 1, 0, 0, 0])
    d, paths = dtw.warping_paths(s1, s2)
    # print(d, "\n", paths)
    dtwvis.plot_warpingpaths(s1, s2, paths, filename=directory / "temp1.png")

    weights = np.full((len(s1), 8), np.inf)
    weights[:, 2:4] = 0.0
    weights[4:7, 2:4] = 10.0
    weights[:, 4:6] = 0.0
    weights[4:7, 4:6] = 10.0
    print(weights)
    d, paths = dtww.warping_paths(s1, s2, weights)
    # print(d, "\n", paths)
    dtwvis.plot_warpingpaths(s1, s2, paths, filename=directory / "temp2.png")


def test_distance2(directory=None):
    directory = prepare_directory(directory)
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
        plot_series(s, l)

    prototypeidx = 0
    ml_values, cl_values, clfs = dtww.series_to_dt(s, l, prototypeidx, max_clfs=5, savefig=str(directory / "dts.dot"))
    logger.debug(f"ml_values = {dict(ml_values)}")
    logger.debug(f"cl_values = {dict(cl_values)}")
    weights = dtww.compute_weights_from_mlclvalues(s[prototypeidx], ml_values, cl_values, only_max=False, strict_cl=True)

    if directory:
        plot_margins(s[prototypeidx], weights, clfs)


def test_distance3(directory=None):
    directory = prepare_directory(directory)
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
    if directory:
        wp_fn = directory / "warping_paths.png"
        dtwvis.plot_warpingpaths(s[0], s[1], paths, path, filename=wp_fn)


def test_distance4(directory=None):
    directory = prepare_directory(directory)
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
        plot_series(s, l)

    prototypeidx = 0
    ml_values, cl_values, clf = dtww.series_to_dt(s, l, prototypeidx, window=2)
    logger.debug(f"ml_values = {dict(ml_values)}")
    logger.debug(f"cl_values = {dict(cl_values)}")
    weights = dtww.compute_weights_from_mlclvalues(s[prototypeidx], ml_values, cl_values,
                                                   only_max=False, strict_cl=True)
    if directory:
        plot_margins(s[prototypeidx], weights, clf)


def test_distance5(directory=None):
    directory = prepare_directory(directory)
    s = np.array([
        [0., 0, 0, 2,  0, -2, 0, 0,  0, 0, 0, 0,  0, 0, 0],  # 0
        [0., 0, 2, 0, -2,  0, 2, 0, -2, 0, 2, 0, -2, 0, 0],  # 1
        [0., 0, 2, 0,  0,  0, 2, 0,  0, 0, 2, 0,  0, 0, 0]   # 2
    ])
    l = np.array([1, 1, 0])

    if directory:
        plot_series(s, l)

    prototypeidx = 0
    ml_values, cl_values, clf = dtww.series_to_dt(s, l, prototypeidx, window=4)
    logger.debug(f"ml_values = {dict(ml_values)}")
    logger.debug(f"cl_values = {dict(cl_values)}")
    weights = dtww.compute_weights_from_mlclvalues(s[prototypeidx], ml_values, cl_values,
                                                   only_max=False, strict_cl=True)
    if directory:
        plot_margins(s[prototypeidx], weights, clf)


if __name__ == "__main__":
    # Print options
    np.set_printoptions(precision=2)
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
    # test_distance1(directory=directory)
    test_distance2(directory=directory)
    # test_distance3(directory=directory)
    # test_distance4(directory=directory)
    # test_distance5(directory=directory)
