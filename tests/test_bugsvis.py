import os
import sys
import math
import tempfile
import pytest
import logging
from pathlib import Path

from dtaidistance import dtw, clustering, util_numpy
import dtaidistance.dtw_visualisation as dtwvis


directory = None
logger = logging.getLogger("be.kuleuven.dtai.distance")
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")
scipyonly = pytest.mark.skipif("util_numpy.test_without_scipy()")


@scipyonly
@numpyonly
def test_bug1():
    with util_numpy.test_uses_numpy() as np:
        series = np.array([
            [0., 0, 1, 2, 1, 0, 1, 0, 0],
            [0., 1, 2, 0, 0, 0, 0, 0, 0],
            [1., 2, 0, 0, 0, 0, 0, 1, 1],
            [0., 0, 1, 2, 1, 0, 1, 0, 0],
            [0., 1, 2, 0, 0, 0, 0, 0, 0],
            [1., 2, 0, 0, 0, 0, 0, 1, 1]])
        model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
        cluster_idx = model.fit(series)

        if directory:
            hierarchy_fn = directory / "hierarchy.png"
        else:
            file = tempfile.NamedTemporaryFile()
            hierarchy_fn = Path(file.name + "_hierarchy.png")
        if not dtwvis.test_without_visualization():
            model.plot(hierarchy_fn)
            print("Figure saved to", hierarchy_fn)


@numpyonly
def test_bug2():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double)
        s2 = np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0])
        d1a = dtw.distance_fast(s1, s2, window=2)
        d1b = dtw.distance(s1, s2, window=2)

        if directory:
            fn = directory / "warpingpaths.png"
        else:
            file = tempfile.NamedTemporaryFile()
            fn = Path(file.name + "_warpingpaths.png")
        d2, paths = dtw.warping_paths(s1, s2, window=2)
        best_path = dtw.best_path(paths)
        if not dtwvis.test_without_visualization():
            dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename=fn, shownumbers=False)
            print("Figure saved to", fn)

        assert d1a == pytest.approx(d2)
        assert d1b == pytest.approx(d2)


@scipyonly
@numpyonly
def test_bug3():
    with util_numpy.test_uses_numpy() as np:
        series = np.array([
            np.array([1, 2, 1]),
            np.array([0., 1, 2, 0, 0, 0, 0, 0, 0]),
            np.array([1., 2, 0, 0, 0, 0, 0, 1, 1, 3, 4, 5]),
            np.array([0., 0, 1, 2, 1, 0, 1]),
            np.array([0., 1, 2, 0, 0, 0, 0, 0]),
            np.array([1., 2, 0, 0, 0, 0, 0, 1, 1])], dtype=object)
        ds = dtw.distance_matrix(series)
        print(ds)

        model = clustering.LinkageTree(dtw.distance_matrix, {})
        cluster_idx = model.fit(series)
        print(cluster_idx)

        if directory:
            fn = directory / "bug3.png"
        else:
            file = tempfile.NamedTemporaryFile()
            fn = Path(file.name + "_bug3.png")

        if not dtwvis.test_without_visualization():
            model.plot(fn, show_ts_label=True)


@numpyonly
def test_bug4():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
        s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
        path = dtw.warping_path(s1, s2)

        if directory:
            fn = directory / "bug4.png"
        else:
            file = tempfile.NamedTemporaryFile()
            fn = Path(file.name + "_bug4.png")

        if not dtwvis.test_without_visualization():
            dtwvis.plot_warping(s1, s2, path, filename=str(fn))


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = 0
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print("Saving files to {}".format(directory))
    # test_bug1()
    test_bug2()
    # test_bug3()
    # test_bug4()
