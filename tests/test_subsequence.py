import pytest
import os
import random
import time
from pathlib import Path

from dtaidistance import util_numpy
from dtaidistance.subsequence.dtw import subsequence_alignment, local_concurrences, subsequence_search
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance.exceptions import MatplotlibException

directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_dtw_subseq1():
    with util_numpy.test_uses_numpy() as np:
        query = np.array([1., 2, 0])
        series = np.array([1., 0, 1, 2, 1, 0, 2, 0, 3, 0, 0])
        sa = subsequence_alignment(query, series)
        mf = sa.matching_function()
        # print(f'{mf=}')
        match = sa.best_match()
        # print(match)
        # print(f'Segment={match.segment}')
        # print(f'Path={match.path}')
        if not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            if directory:
                plt.plot(mf)
                plt.savefig(directory / "subseq_matching.png")
                dtwvis.plot_warpingpaths(query, series, sa.warping_paths(), match.path,
                                         filename=directory / "subseq_warping.png")
                plt.close()
        best_k = sa.kbest_matches(k=3)
        assert match.path == [(0, 2), (1, 3), (2, 4)]
        assert [m.segment for m in best_k] == [[2, 4], [5, 7], [0, 1]]


@numpyonly
def test_dtw_subseq_eeg():
    with util_numpy.test_uses_numpy() as np:
        data_fn = Path(__file__).parent / 'rsrc' / 'EEGRat_10_1000.txt'
        data = np.loadtxt(data_fn)
        series = np.array(data[1500:1700])
        query = np.array(data[1331:1352])

        sa = subsequence_alignment(query, series)
        match = sa.best_match()
        kmatches = list(sa.kbest_matches(k=15, overlap=0))
        segments = [m.segment for m in kmatches]
        segments_sol = [[38, 56], [19, 37], [167, 185], [124, 143], [84, 100], [59, 77], [150, 162], [101, 121], [0, 15]]

        assert segments == segments_sol

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")

            fn = directory / "test_dtw_subseq_eeg1.png"
            fig = plt.figure(figsize=(20, 30))
            dtwvis.plot_warpingpaths(query, series, sa.warping_paths(), match.path, figure=fig)
            plt.savefig(fn)
            plt.close(fig)

            fn = directory / "test_dtw_subseq_eeg2.png"
            startidx, endidx = match.segment
            fig = plt.figure()
            plt.plot(query, label='query')
            plt.plot(series[startidx: endidx], label='best match')
            plt.legend()
            plt.savefig(fn)
            plt.close(fig)

            fn = directory / "test_dtw_subseq_eeg3.png"
            fig = plt.figure(figsize=(20, 10))
            fig, ax = dtwvis.plot_warpingpaths(query, series, sa.warping_paths(), path=-1, figure=fig)
            print(f'plotting {len(kmatches)} matches')
            for kmatch in kmatches:
                dtwvis.plot_warpingpaths_addpath(ax, kmatch.path)
            plt.savefig(fn)
            plt.close(fig)


@numpyonly
def test_dtw_subseq_bug1():
    use_c = True
    with util_numpy.test_uses_numpy() as np:
        query = np.array([-0.86271501, -1.32160597, -1.2307838, -0.97743775, -0.88183547,
                          -0.71453147, -0.70975136, -0.65238999, -0.48508599, -0.40860416,
                          -0.5567877, -0.39904393, -0.51854679, -0.51854679, -0.23652005,
                          -0.21261948, 0.16978966, 0.21281068, 0.6573613, 1.28355626,
                          1.88585065, 1.565583, 1.40305912, 1.64206483, 1.8667302])
        s1 = np.array([-0.87446789, 0.50009064, -1.43396157, 0.52081263, 1.28752619])
        s2 = np.array([1.19125347, 0.78778189, -0.95770272, -1.02133264])
        sa = subsequence_alignment(query, s1, use_c=use_c)
        assert sa.best_match().value == pytest.approx(0.08735692337954708)
        sa = subsequence_alignment(query, s2, use_c=use_c)
        assert sa.best_match().value == pytest.approx(0.25535859535443606)


@numpyonly
def test_dtw_localconcurrences_eeg():
    with util_numpy.test_uses_numpy() as np:
        data_fn = Path(__file__).parent / 'rsrc' / 'EEGRat_10_1000.txt'
        data = np.loadtxt(data_fn)
        series = np.array(data[1500:1700])

        gamma = 1
        # domain = 2 * np.std(series)
        # affinity = np.exp(-gamma * series)
        # print(f'Affinity in [{np.min(affinity)}, {np.max(affinity)}]\n'
        #       f'             {np.mean(affinity)} +- {np.std(affinity)}\n'
        #       f'             {np.exp(-gamma * np.mean(series))} +- {np.exp(-gamma * np.std(series))}\n'
        #       f'             {np.exp(-gamma * np.percentile(series, 75))} / {np.exp(-gamma * np.median(series))} / {np.exp(-gamma * np.percentile(series, 25))}\n')
        tau_stddev = 0.40
        diffp = tau_stddev*np.std(series)
        delta = -2 * np.exp(-gamma * diffp**2)  # -len(series)/2  # penalty
        delta_factor = 0.5
        tau = np.exp(-gamma * diffp**2)  # threshold
        print(f'{tau=}, {delta=}')
        # tau=0.8532234738897421, delta=-1.7064469477794841
        buffer = 10
        minlen = 20
        lc = local_concurrences(series, gamma=gamma, tau=tau, delta=delta, delta_factor=delta_factor)
        print(f'{lc.tau=}, {lc.delta=}')
        matches = []
        for match in lc.kbest_matches(k=100, minlen=minlen, buffer=buffer):
            if match is None:
                break
            matches.append(match)
        print( [(m.row, m.col) for m in matches])
        # assert [(m.row, m.col) for m in matches] == [(84, 95), (65, 93), (50, 117), (117, 200), (32, 180),
        #                                              (160, 178), (96, 139), (138, 181), (71, 200), (71, 117),
        #                                              (73, 137), (52, 138), (12, 117), (117, 178), (117, 160),
        #                                              (30, 160), (32, 52), (30, 117), (117, 135), (160, 200),
        #                                              (178, 200), (11, 52), (71, 160), (134, 160), (135, 200),
        #                                              (30, 200), (50, 200), (11, 73), (50, 160), (12, 33), (11, 137),
        #                                              (36, 143), (11, 179), (88, 160), (66, 178), (11, 93)]

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fn = directory / "test_dtw_localconcurrences.png"
            fig = plt.figure()
            fig, ax = dtwvis.plot_warpingpaths(series, series, lc.wp, path=-1, figure=fig)
            for match in matches:
                dtwvis.plot_warpingpaths_addpath(ax, match.path)
            plt.savefig(fn)
            plt.close(fig)


@numpyonly
def test_dtw_localconcurrences_short():
    with util_numpy.test_uses_numpy() as np:
        series = np.array([0, -1, -1, 0, 1, 2, 1, 0, 0, 0, 1, 3, 2, 1, 0, 0, 0, -1, 0])

        gamma = 1
        threshold_tau = 70
        delta = -2 * np.exp(-gamma * np.percentile(series, threshold_tau))  # -len(series)/2  # penalty
        delta_factor = 0.5
        tau = np.exp(-gamma * np.percentile(series, threshold_tau))  # threshold
        # print(f'{tau=}, {delta=}')
        buffer = 10
        minlen = 3
        lc = local_concurrences(series, gamma=gamma, tau=tau, delta=delta, delta_factor=delta_factor, penalty=1)
        matches = []
        for match in lc.kbest_matches(k=100, minlen=minlen, buffer=buffer):
            if match is None:
                break
            matches.append(match)

        assert [(m.row, m.col) for m in matches] == [(10, 17), (4, 19)]

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fn = directory / "test_dtw_localconcurrences_short.png"
            fig = plt.figure()
            fig, ax = dtwvis.plot_warpingpaths(series, series, lc.wp, path=-1, figure=fig)
            for match in matches:
                dtwvis.plot_warpingpaths_addpath(ax, match.path)
            plt.savefig(fn)
            plt.close(fig)


@numpyonly
def test_dtw_subseqsearch_eeg():
    with util_numpy.test_uses_numpy() as np:
        data_fn = Path(__file__).parent / 'rsrc' / 'EEGRat_10_1000.txt'
        data = np.loadtxt(data_fn)
        series = np.array(data[1500:1700])
        query = np.array(data[1331:1352])
        # print(f'{len(series)=}')

        k = 3
        s = []
        s_idx = []
        w = 22  # window size
        ws = int(np.floor(w/2))  # shift size
        wn = int(np.floor((len(series) - (w - ws)) / ws))
        si, ei = 0, w
        for i in range(wn):
            s.append(series[si:ei])
            s_idx.append(si)
            si += ws
            ei += ws
        tic = time.perf_counter()
        sa = subsequence_search(query, s)
        best = sa.kbest_matches(k=k)
        toc = time.perf_counter()
        print(f"Searching performed in {toc - tic:0.4f} seconds")
        # print(sa.distances)
        # print(best)

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
                from matplotlib import gridspec
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fn = directory / "test_dtw_subseqsearch_eeg.png"
            ymin, ymax = np.min(series), np.max(series)
            fig = plt.figure()
            if k is None:
                k = len(s)
            gs = gridspec.GridSpec(3, k, wspace=0.5, hspace=1)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(query)
            ax.set_title('Query')
            ax.set_ylim((ymin, ymax))
            for idx, match in enumerate(best):
                ax = fig.add_subplot(gs[1, idx])
                if idx == 0:
                    ax.set_title(f'Best {k} windows')
                ax.set_ylim((ymin, ymax))
                ax.plot(s[match.idx])
            ax = fig.add_subplot(gs[2, :])
            ax.set_ylim((ymin, ymax))
            ax.set_title(f'Series with windows')
            for idx in s_idx:
                ax.vlines(idx, ymin, ymax, color='grey', alpha=0.4)
            ax.plot(series)
            for idx, match in enumerate(best):
                ax.vlines(s_idx[match.idx], ymin, ymax, color='red')
            plt.savefig(fn)
            plt.close(fig)


if __name__ == "__main__":
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    with util_numpy.test_uses_numpy() as np:
        np.set_printoptions(precision=6, linewidth=120)
        # test_dtw_subseq1()
        # test_dtw_subseq_eeg()
        test_dtw_subseq_bug1()
        # test_dtw_localconcurrences_eeg()
        # test_dtw_subseqsearch_eeg()
        # test_dtw_localconcurrences_short()
