import os
import sys
import tempfile
import pytest
import logging
from pathlib import Path

from dtaidistance import dtw, dtw_ndim, clustering, util_numpy
import dtaidistance.dtw_visualisation as dtwvis
from dtaidistance.exceptions import PyClusteringException


logger = logging.getLogger("be.kuleuven.dtai.distance")
directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")
scipyonly = pytest.mark.skipif("util_numpy.test_without_scipy()")


@numpyonly
def test_clustering():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1]])

        def test_hook(from_idx, to_idx, distance):
            assert (from_idx, to_idx) in [(3, 0), (4, 1), (5, 2), (1, 0)]
        model = clustering.Hierarchical(dtw.distance_matrix_fast, {}, 2, merge_hook=test_hook,
                                        show_progress=False)
        cluster_idx = model.fit(s)
        assert cluster_idx[0] == {0, 1, 3, 4}
        assert cluster_idx[2] == {2, 5}


@numpyonly
def test_clustering_tree():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [1., 2, 0, 0, 0, 0, 0, 1, 1]])

        def test_hook(from_idx, to_idx, distance):
            assert (from_idx, to_idx) in [(3, 0), (4, 1), (5, 2), (6, 2), (1, 0), (2, 0)]
        model = clustering.Hierarchical(dtw.distance_matrix_fast, {}, merge_hook=test_hook,
                                        show_progress=False)
        modelw = clustering.HierarchicalTree(model)
        cluster_idx = modelw.fit(s)
        assert cluster_idx[0] == {0, 1, 2, 3, 4, 5, 6}

        if directory:
            hierarchy_fn = os.path.join(directory, "hierarchy.png")
            graphviz_fn = os.path.join(directory, "hierarchy.dot")
        else:
            file = tempfile.NamedTemporaryFile()
            hierarchy_fn = file.name + "_hierarchy.png"
            graphviz_fn = file.name + "_hierarchy.dot"

        if not dtwvis.test_without_visualization():
            modelw.plot(hierarchy_fn)
            print("Figure saved to", hierarchy_fn)

        with open(graphviz_fn, "w") as ofile:
            print(modelw.to_dot(), file=ofile)
        print("Dot saved to", graphviz_fn)


@numpyonly
def test_clustering_tree_ndim():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
             [[0.,0.], [0,0], [1,0], [2,0], [1,0], [0,0], [1,0], [0,0], [0,0]],
             [[0.,0.], [1,0], [2,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]],
             [[1.,0.], [2,0], [0,0], [0,0], [0,0], [0,0], [0,0], [1,0], [1,0]]])

        model = clustering.Hierarchical(dtw_ndim.distance_matrix_fast, {'ndim':2},
                                        show_progress=False)
        cluster_idx = model.fit(s)
        assert cluster_idx[0] == {0, 1, 2}


@numpyonly
def test_kmeans_ndim():
    with util_numpy.test_uses_numpy() as np:
        np.random.seed(seed=3980)
        arr = np.random.random((10, 10, 3))

        model = clustering.kmeans.KMeans(k=2, dists_options={"use_c": True})
        cl, p = model.fit(arr)
        assert str(cl) == "{0: {1, 2, 4, 6}, 1: {0, 3, 5, 7, 8, 9}}"


@numpyonly
def test_kmeans_ndim2():
    with util_numpy.test_uses_numpy() as np:
        np.random.seed(seed=3980)
        arr = np.random.random((10, 10, 3))

        model = clustering.kmeans.KMeans(k=2, dists_options={"use_c": False})
        cl, p = model.fit(arr, use_parallel=False)
        assert str(cl) == "{0: {1, 2, 4, 6}, 1: {0, 3, 5, 7, 8, 9}}"


@numpyonly
def test_clustering_tree_maxdist():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [1., 2, 0, 0, 0, 0, 0, 1, 1]])

        def test_hook(from_idx, to_idx, distance):
            assert (from_idx, to_idx) in [(3, 0), (4, 1), (5, 2), (6, 2), (1, 0), (2, 0)]
        model = clustering.Hierarchical(dtw.distance_matrix_fast, {}, merge_hook=test_hook,
                                        show_progress=False, max_dist=0.1)
        modelw = clustering.HierarchicalTree(model)
        cluster_idx = modelw.fit(s)
        assert cluster_idx[0] == {0, 1, 2, 3, 4, 5, 6}

        if directory:
            hierarchy_fn = os.path.join(directory, "hierarchy.png")
            graphviz_fn = os.path.join(directory, "hierarchy.dot")
        else:
            file = tempfile.NamedTemporaryFile()
            hierarchy_fn = file.name + "_hierarchy.png"
            graphviz_fn = file.name + "_hierarchy.dot"

        if not dtwvis.test_without_visualization():
            modelw.plot(hierarchy_fn)
            print("Figure saved to", hierarchy_fn)

        with open(graphviz_fn, "w") as ofile:
            print(modelw.to_dot(), file=ofile)
        print("Dot saved to", graphviz_fn)


@scipyonly
@numpyonly
def test_linkage_tree():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [1., 2, 0, 0, 0, 0, 0, 1, 1]])

        model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
        cluster_idx = model.fit(s)

        if directory:
            hierarchy_fn = os.path.join(directory, "hierarchy.png")
            graphviz_fn = os.path.join(directory, "hierarchy.dot")
        else:
            file = tempfile.NamedTemporaryFile()
            hierarchy_fn = file.name + "_hierarchy.png"
            graphviz_fn = file.name + "_hierarchy.dot"
        if not dtwvis.test_without_visualization():
            model.plot(hierarchy_fn)
            print("Figure saved to", hierarchy_fn)
        with open(graphviz_fn, "w") as ofile:
            print(model.to_dot(), file=ofile)
        print("Dot saved to", graphviz_fn)


@numpyonly
def test_trace_hierarchical():
    with util_numpy.test_uses_numpy() as np, util_numpy.test_uses_scipy() as scipy:
        nb = 20
        rsrc_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'Trace_TRAIN.txt')
        data = np.loadtxt(rsrc_fn)
        labels = data[:, 0]
        series = data[:, 1:]
        model = clustering.LinkageTree(dtw.distance_matrix_fast, {'parallel': True})
        model.fit(series[:nb])

        if not dtwvis.test_without_visualization() and directory:
            import matplotlib.pyplot as plt
            hierarchy_fn = os.path.join(directory, "trace_hierarchical.png")
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
            plt.rcParams['font.family'] = 'monospace'
            ts_label = lambda idx: f'{int(idx):>2}={int(labels[idx])}'
            model.plot(hierarchy_fn, axes=ax, show_ts_label=ts_label, ts_label_margin=300)


@scipyonly
@numpyonly
def test_controlchart():
    with util_numpy.test_uses_numpy() as np:
        series = np.zeros((600, 60))
        rsrc_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'synthetic_control.data')
        with open(rsrc_fn, 'r') as ifile:
            for idx, line in enumerate(ifile.readlines()):
                series[idx, :] = line.split()
        s = []
        for idx in range(0, 600, 20):
            s.append(series[idx, :])

        model = clustering.LinkageTree(dtw.distance_matrix_fast, {'parallel': True})
        cluster_idx = model.fit(s)

        if not dtwvis.test_without_visualization():
            import matplotlib.pyplot as plt
            if directory:
                hierarchy_fn = os.path.join(directory, "hierarchy.png")
            else:
                file = tempfile.NamedTemporaryFile()
                hierarchy_fn = file.name + "_hierarchy.png"
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
            show_ts_label = lambda idx: "ts-" + str(idx)
            # show_ts_label = list(range(len(s)))

            def curcmap(idx):
                if idx % 2 == 0:
                    return 'r'
                return 'g'

            model.plot(hierarchy_fn, axes=ax, show_ts_label=show_ts_label,
                       show_tr_label=True, ts_label_margin=-10,
                       ts_left_margin=10, ts_sample_length=1, ts_color=curcmap)
            print("Figure saved to", hierarchy_fn)


@scipyonly
@numpyonly
def test_plotbug1():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
        s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0])

        series = s1, s2

        m = clustering.LinkageTree(dtw.distance_matrix, {})
        m.fit(series)

        if not dtwvis.test_without_visualization():
            if directory:
                hierarchy_fn = os.path.join(directory, "clustering.png")
            else:
                file = tempfile.NamedTemporaryFile()
                hierarchy_fn = file.name + "_clustering.png"
            m.plot(hierarchy_fn)
            print("Figure save to", hierarchy_fn)


@numpyonly
def test_clustering_centroid():
    with util_numpy.test_uses_numpy() as np:
        s = np.array([
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [1., 2, 0, 0, 0, 0, 0, 1, 1]])

        # def test_hook(from_idx, to_idx, distance):
        #     assert (from_idx, to_idx) in [(3, 0), (4, 1), (5, 2), (6, 2), (1, 0), (2, 0)]
        model = clustering.KMedoids(dtw.distance_matrix_fast, {}, k=3,
                                    show_progress=False)
        try:
            cluster_idx = model.fit(s)
        except PyClusteringException:
            return
        # assert cluster_idx[0] == {0, 1, 2, 3, 4, 5, 6}

        if not dtwvis.test_without_visualization():
            if directory:
                png_fn = os.path.join(directory, "centroid.png")
            else:
                file = tempfile.NamedTemporaryFile()
                png_fn = file.name + "_centroid.png"
            model.plot(png_fn)
            print("Figure saved to", png_fn)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print("Saving files to {}".format(directory))
    # test_clustering_tree()
    # test_clustering_tree_ndim()
    # test_clustering_tree_maxdist()
    # test_linkage_tree()
    test_trace_hierarchical()
    # test_controlchart()
    # test_plotbug1()
    # test_clustering_centroid()
