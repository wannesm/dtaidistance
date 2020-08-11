import os
import sys
import tempfile
import pytest
import logging
from pathlib import Path
from dtaidistance import dtw, clustering, util_numpy


logger = logging.getLogger("be.kuleuven.dtai.distance")
directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


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
        modelw.plot(hierarchy_fn)
        print("Figure saved to", hierarchy_fn)
        with open(graphviz_fn, "w") as ofile:
            print(modelw.to_dot(), file=ofile)
        print("Dot saved to", graphviz_fn)


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
        modelw.plot(hierarchy_fn)
        print("Figure saved to", hierarchy_fn)
        with open(graphviz_fn, "w") as ofile:
            print(modelw.to_dot(), file=ofile)
        print("Dot saved to", graphviz_fn)


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
        model.plot(hierarchy_fn)
        print("Figure saved to", hierarchy_fn)
        with open(graphviz_fn, "w") as ofile:
            print(model.to_dot(), file=ofile)
        print("Dot saved to", graphviz_fn)


@numpyonly
def test_controlchart():
    with util_numpy.test_uses_numpy() as np:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
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


@numpyonly
def test_plotbug1():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
        s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0])

        series = s1, s2

        m = clustering.LinkageTree(dtw.distance_matrix, {})
        m.fit(series)

        if directory:
            hierarchy_fn = os.path.join(directory, "clustering.png")
        else:
            file = tempfile.NamedTemporaryFile()
            hierarchy_fn = file.name + "_clustering.png"
        m.plot(hierarchy_fn)
        print("Figure save to", hierarchy_fn)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_clustering_tree()
    test_clustering_tree_maxdist()
    # test_linkage_tree()
    # test_controlchart()
    # test_plotbug1()
