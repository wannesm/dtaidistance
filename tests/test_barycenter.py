import os
import sys
import time
import pytest
import logging
from pathlib import Path

from dtaidistance import dtw_barycenter, util_numpy
import dtaidistance.dtw_visualisation as dtwvis
from dtaidistance.exceptions import MatplotlibException, PyClusteringException
from dtaidistance.clustering.kmeans import KMeans
from dtaidistance.dtw_barycenter import dba_loop
from dtaidistance.preprocessing import differencing


logger = logging.getLogger("be.kuleuven.dtai.distance")
directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")
scipyonly = pytest.mark.skipif("util_numpy.test_without_scipy()")


@pytest.mark.skip
@numpyonly
def test_pair():

    with util_numpy.test_uses_numpy() as np:
        s = np.array([
            [0.5, 1, 2, 3, 2.0, 2.1, 1.0, 0, 0, 0],
            [0.4, 0, 1, 1.5, 1.9, 2.0, 0.9, 1, 0, 0]
        ])

        t = s.shape[1]
        # c = np.array([0.0, 0.5, 1.5, 2.5, 2, 2, 1, 0.5, 0, 0])
        # c = np.zeros((s.shape[1],))
        c = s[0, :]
        max_it = 1
        avgs = [c]

        tic = time.perf_counter()
        avg, avgs = dtw_barycenter.dba_loop(s, c, max_it=max_it, thr=0.0001, keep_averages=True,
                                            use_c=False)
        toc = time.perf_counter()
        print(f'DBA_loop: {toc - tic:0.4f} sec')
        print(avg)

        # tic = time.perf_counter()
        # for it in range(max_it):
        #     avg = dtw_barycenter.dba(s, c, use_c=True)
        #     avgs.append(avg)
        #     c = avg
        # toc = time.perf_counter()
        # print(f'DBA: {toc - tic:0.4f} sec')

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fig, ax = plt.subplots(nrows=max_it, ncols=1)
            fn = directory / "test_pair_barycenter.png"
            for it in range(len(avgs)):
                dtwvis.plot_average(s[0, :], s[1, :], avgs[it], None, None, ax=ax[it])
                ax[it].set_title(f'Iteration {it}')
            fig.savefig(str(fn))
            plt.close()


@numpyonly
def test_trace():
    with util_numpy.test_uses_numpy() as np:
        rsrc_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'Trace_TRAIN.txt')
        data = np.loadtxt(rsrc_fn)
        labels = data[:, 0]
        # series = data[:, 1:]
        series = data[labels == 1, 1:][:2, :].copy()
        # c = series[0, :]
        print(type(series))
        print(series.shape)

        tic = time.perf_counter()
        avg = dtw_barycenter.dba_loop(series, c=None, max_it=100, thr=0.000001,
                                      nb_initial_samples=4, use_c=True)
        toc = time.perf_counter()
        print(f'DBA: {toc - tic:0.4f} sec')

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
            fn = directory / "test_trace_barycenter.png"

            for serie in series:
                ax[0].plot(serie, alpha=0.5)
            ax[1].plot(avg)

            fig.savefig(str(fn))
            plt.close()


@numpyonly
def test_trace_mask():
    with util_numpy.test_uses_numpy() as np:
        rsrc_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'Trace_TRAIN.txt')
        data = np.loadtxt(rsrc_fn)
        labels = data[:, 0]
        series = data[:, 1:]
        mask = np.full((len(labels),), False, dtype=bool)
        mask[:] = (labels == 1)
        # c = series[0, :]
        print(type(series))
        print(series.shape)

        tic = time.perf_counter()
        avg = dtw_barycenter.dba_loop(series, c=None, max_it=100, thr=0.000001, mask=mask, use_c=True)
        toc = time.perf_counter()
        print(f'DBA: {toc - tic:0.4f} sec')

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
            fn = directory / "test_trace_barycenter.png"

            for idx, serie in enumerate(series):
                if mask[idx]:
                    ax[0].plot(serie, alpha=0.5)
            ax[1].plot(avg)

            fig.savefig(str(fn))
            plt.close()


@numpyonly
def test_trace_kmeans():
    with util_numpy.test_uses_numpy() as np:
        k = 4
        max_it = 10
        max_dba_it = 20
        rsrc_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'Trace_TRAIN.txt')
        data = np.loadtxt(rsrc_fn)
        labels = data[:, 0]
        series = data[:, 1:]
        mask = np.full((len(labels),), False, dtype=bool)
        mask[:] = (labels == 1)
        # c = series[0, :]
        print(type(series))
        print(series.shape)
        window = int(series.shape[1] * 0.5)

        # Z-normalize sequences
        series = (series - series.mean(axis=1)[:, None]) / series.std(axis=1)[:, None]

        # Align start and/or end values
        # avg_start = series[:, :20].mean(axis=1)
        # avg_end = series[:, 20:].mean(axis=1)
        # series = (series - avg_start[:, None])

        # Perform k-means
        tic = time.perf_counter()
        model = KMeans(k=k, max_it=max_it, max_dba_it=max_dba_it, drop_stddev=1,
                       nb_prob_samples=0,
                       dists_options={"window": window},
                       initialize_with_kmedoids=False,
                       initialize_with_kmeanspp=True)
        try:
            cluster_idx, performed_it = model.fit(series, use_c=True, use_parallel=False)
        except PyClusteringException:
            return
        toc = time.perf_counter()
        print(f'DBA ({performed_it} iterations: {toc - tic:0.4f} sec')

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fig, ax = plt.subplots(nrows=k, ncols=2, figsize=(10, 4),
                                   sharex='all', sharey='all')
            fn = directory / "test_trace_barycenter.png"

            all_idx = set()
            for ki in range(k):
                ax[ki, 0].plot(model.means[ki])
                for idx in cluster_idx[ki]:
                    ax[ki, 1].plot(series[idx], alpha=0.3)
                    if idx in all_idx:
                        raise Exception(f'Series in multiple clusters: {idx}')
                    all_idx.add(idx)
            assert(len(all_idx) == len(series))
            fig.savefig(str(fn))
            plt.close()

            fig, ax = plt.subplots(nrows=k, ncols=1, figsize=(5, 4),
                                   sharex='all', sharey='all')
            fn = directory / "test_trace_barycenter_solution.png"
            for i in range(len(labels)):
                ax[int(labels[i]) - 1].plot(series[i], alpha=0.3)
            fig.savefig(str(fn))
            plt.close()


@numpyonly
@scipyonly
def test_trace_kmeans_differencing():
    with util_numpy.test_uses_numpy() as np, util_numpy.test_uses_scipy() as scipy:
        k = 4
        max_it = 10
        max_dba_it = 20
        nb_prob_samples = 0
        use_c = True
        rsrc_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'Trace_TRAIN.txt')
        data = np.loadtxt(rsrc_fn)
        labels = data[:, 0]
        series = data[:, 1:]
        mask = np.full((len(labels),), False, dtype=bool)
        mask[:] = (labels == 1)
        # c = series[0, :]
        print(type(series))
        print(series.shape)
        window = int(series.shape[1] * 0.5)

        # Differencing
        # The baseline differences are not relevant thus we cluster based
        # on the result of differencing.
        # Also the high-freq noise dominates the local differences, thus
        # we apply a low-pass filter first.
        series_orig = series.copy()
        # signal = scipy.import_signal()
        # series = np.diff(series, n=1, axis=1)
        # fs = 100  # sample rate, Hz
        # cutoff = 10  # cut off frequency, Hz
        # nyq = 0.5 * fs  # Nyquist frequency
        # b, a = signal.butter(2, cutoff / nyq, btype='low', analog=False, output='ba')
        # series = signal.filtfilt(b, a, series, axis=1)
        series = differencing(series, smooth=0.1)

        # Perform k-means
        tic = time.perf_counter()
        model = KMeans(k=k, max_it=max_it, max_dba_it=max_dba_it, drop_stddev=1,
                       nb_prob_samples=nb_prob_samples,
                       dists_options={"window": window},
                       initialize_with_kmedoids=False,
                       initialize_with_kmeanspp=True)
        try:
            cluster_idx, performed_it = model.fit(series, use_c=use_c, use_parallel=False)
        except PyClusteringException:
            return
        toc = time.perf_counter()
        print(f'DBA ({performed_it} iterations: {toc - tic:0.4f} sec')

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fig, ax = plt.subplots(nrows=k, ncols=3, figsize=(10, 4),
                                   sharex='all', sharey='all')
            fn = directory / "test_trace_barycenter.png"

            all_idx = set()
            mask = np.full((k, len(series_orig)), False, dtype=bool)
            for ki in range(k):
                ax[ki, 0].plot(model.means[ki])
                for idx in cluster_idx[ki]:
                    ax[ki, 2].plot(series_orig[idx], alpha=0.3)
                    mask[ki, idx] = True
                    if idx in all_idx:
                        raise Exception(f'Series in multiple clusters: {idx}')
                    all_idx.add(idx)

            series_orig = (series_orig - series_orig.mean(axis=1)[:, None]) / series_orig.std(axis=1)[:, None]
            for ki, mean in enumerate(model.means):
                # dba = dba_loop(series_orig, c=None, mask=mask[ki, :],
                #                max_it=max_it, thr=None, use_c=use_c,
                #                nb_prob_samples=nb_prob_samples)
                print(mean.shape)
                dba = np.r_[0, mean].cumsum()
                ax[ki, 1].plot(dba)
            assert(len(all_idx) == len(series))
            ax[0, 0].set_title("DBA Differencing + LP")
            ax[0, 1].set_title("DBA Original series")
            ax[0, 2].set_title("Clustered series")
            fig.savefig(str(fn))
            plt.close()

            fig, ax = plt.subplots(nrows=k, ncols=1, figsize=(5, 4),
                                   sharex='all', sharey='all')
            fn = directory / "test_trace_barycenter_solution.png"
            for i in range(len(labels)):
                ax[int(labels[i]) - 1].plot(series_orig[i], alpha=0.3)
            fig.savefig(str(fn))
            plt.close()


@numpyonly
def test_nparray_kmeans():
    with util_numpy.test_uses_numpy() as np:
        k = 4
        max_it = 10
        max_dba_it = 20

        series = np.array(
            [[0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1],
             [0., 0, 1, 2, 1, 0, 1, 0, 0],
             [0., 1, 2, 0, 0, 0, 0, 0, 0],
             [1., 2, 0, 0, 0, 0, 0, 1, 1]]
        )
        print(type(series))
        print(series.shape)
        window = int(series.shape[1] * 1.0)

        # Perform k-means
        tic = time.perf_counter()
        model = KMeans(k=k, max_it=max_it, max_dba_it=max_dba_it,
                       dists_options={"window": window},
                       initialize_with_kmedoids=False,
                       initialize_with_kmeanspp=True)
        cluster_idx, performed_it = model.fit(series, use_c=True, use_parallel=False)
        toc = time.perf_counter()
        print(f'DBA ({performed_it} iterations: {toc - tic:0.4f} sec')

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fig, ax = plt.subplots(nrows=k, ncols=2, figsize=(10, 4),
                                   sharex='all', sharey='all')
            fn = directory / "test_nparray_barycenter.png"

            all_idx = set()
            for ki in range(k):
                ax[ki, 0].plot(model.means[ki])
                for idx in cluster_idx[ki]:
                    ax[ki, 1].plot(series[idx], alpha=0.3)
                    if idx in all_idx:
                        raise Exception(f'Series in multiple clusters: {idx}')
                    all_idx.add(idx)
            assert(len(all_idx) == len(series))
            fig.savefig(str(fn))
            plt.close()


@pytest.mark.skip("Not yet implemented")
@numpyonly
def test_ndim_kmeans():
    with util_numpy.test_uses_numpy() as np:
        k = 4
        max_it = 10
        max_dba_it = 20
        # series = np.array(
        #     [[[0., 0], [1, 2], [1, 0], [1, 0]],
        #      [[0., 1], [2, 0], [0, 0], [0, 0]],
        #      [[1., 2], [0, 0], [0, 0], [0, 1]],
        #      [[0., 0], [1, 2], [1, 0], [1, 0]],
        #      [[0., 1], [2, 0], [0, 0], [0, 0]],
        #      [[1., 2], [0, 0], [0, 0], [0, 1]]])
        series = [np.array([[0., 0], [1, 2], [1, 0], [1, 0]]),
             np.array([[0., 1], [2, 0], [0, 0], [0, 0]]),
             np.array([[1., 2], [0, 0], [0, 0], [0, 1]]),
             np.array([[0., 0], [1, 2], [1, 0], [1, 0]]),
             np.array([[0., 1], [2, 0], [0, 0], [0, 0]]),
             np.array([[1., 2], [0, 0], [0, 0], [0, 1]])]
        print(type(series))
        # print(series.shape)
        # window = int(series.shape[1] * 1.0)
        window = None
        print(f'window={window}')

        # Perform k-means
        tic = time.perf_counter()
        model = KMeans(k=k, max_it=max_it, max_dba_it=max_dba_it, drop_stddev=2,
                       dists_options={"window": window},
                       initialize_with_kmedoids=False,
                       initialize_with_kmeanspp=False)
        cluster_idx, performed_it = model.fit(series, use_c=False, use_parallel=False)
        toc = time.perf_counter()
        print(f'DBA ({performed_it} iterations: {toc - tic:0.4f} sec')


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_pair()
    # test_trace()
    # test_trace_mask()
    test_trace_kmeans()
    # test_trace_kmeans_differencing()
    # test_nparray_kmeans()
    # test_ndim_kmeans()
