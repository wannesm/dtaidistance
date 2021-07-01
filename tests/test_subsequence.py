import pytest
import os
import random
from pathlib import Path

from dtaidistance import util_numpy
from dtaidistance.subsequence.dtw import subsequence_search, local_concurrences
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance.exceptions import MatplotlibException

directory = None
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_dtw_subseq1():
    with util_numpy.test_uses_numpy() as np:
        query = np.array([1., 2, 0])
        series = np.array([1., 0, 1, 2, 1, 0, 2, 0, 3, 0, 0])
        sa = subsequence_search(query, series)
        mf = sa.matching_function()
        print(f'{mf=}')
        match = sa.best_match()
        print(match)
        print(f'Segment={match.segment}')
        print(f'Path={match.path}')
        if not dtwvis.test_without_visualization():
            import matplotlib.pyplot as plt
            if directory:
                plt.plot(mf)
                plt.savefig(directory / "subseq_matching.png")
                dtwvis.plot_warpingpaths(query, series, sa.warping_paths(), match.path,
                                         filename=directory / "subseq_warping.png")
                plt.close()
        best_k = sa.kbest_match(k=3)
        print(best_k)


@numpyonly
def test_dtw_localconcurrences():
    data_fn = Path(__file__).parent / 'rsrc' / 'EEGRat_10_1000.txt'
    data = np.loadtxt(data_fn)
    series = np.array(data[1500:1700])

    gamma = 1
    domain = 2 * np.std(series)
    affinity = np.exp(-gamma * series)
    print(f'Affinity in [{np.min(affinity)}, {np.max(affinity)}]\n'
          f'             {np.mean(affinity)} +- {np.std(affinity)}\n'
          f'             {np.exp(-gamma * np.mean(series))} +- {np.exp(-gamma * np.std(series))}\n'
          f'             {np.exp(-gamma * np.percentile(series, 75))} / {np.exp(-gamma * np.median(series))} / {np.exp(-gamma * np.percentile(series, 25))}\n')
    threshold_tau = 70
    delta = -2 * np.exp(-gamma * np.percentile(series, threshold_tau))  # -len(series)/2  # penalty
    delta_factor = 0.5
    tau = np.exp(-gamma * np.percentile(series, threshold_tau))  # threshold
    print(f'{tau=}, {delta=}')
    buffer = 10
    minlen = 20
    lc = local_concurrences(series, gamma=gamma, tau=tau, delta=delta, delta_factor=delta_factor)
    paths = []
    match = lc.next_best_match(minlen=minlen, buffer=buffer)
    if match is not None:
        paths.append(match.path)
    for _ in range(100):
        matchb = lc.next_best_match(minlen=minlen, buffer=buffer)
        if matchb is None:
            break
        paths.append(matchb.path)
    if directory and not dtwvis.test_without_visualization():
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise MatplotlibException("No matplotlib available")
        fn = directory / "test_dtw_localconcurrences.png"
        fig = plt.figure()
        fig, ax = dtwvis.plot_warpingpaths(series, series, lc.wp, path=-1, showlegend=True, figure=fig)
        for path in paths:
            dtwvis.plot_warpingpaths_addpath(ax, path)
        plt.savefig(fn)
        plt.close(fig)


if __name__ == "__main__":
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    with util_numpy.test_uses_numpy() as np:
        np.set_printoptions(precision=6, linewidth=120)
        # test_dtw_subseq1()
        test_dtw_localconcurrences()
