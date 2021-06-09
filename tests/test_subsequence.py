import pytest
import os
import random
from pathlib import Path

from dtaidistance import util_numpy
from dtaidistance.subsequence.dtw import subsequence_search
from dtaidistance import dtw_visualisation as dtwvis


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
        best_match_idx = np.argmin(mf)
        print(f'{best_match_idx=}')
        best_match_start_idx = sa.matching_function_startpoint(best_match_idx)
        print(f'{best_match_start_idx=}')
        best_match_path = sa.matching_function_bestpath(best_match_idx)
        print(f'{best_match_path=}')
        if not dtwvis.test_without_visualization():
            import matplotlib.pyplot as plt
            if directory:
                plt.plot(mf)
                plt.savefig(directory / "subseq_matching.png")
                dtwvis.plot_warpingpaths(query, series, sa.paths, best_match_path, filename=directory / "subseq_warping.png")
                plt.close()


if __name__ == "__main__":
    with util_numpy.test_uses_numpy() as np:
        np.set_printoptions(precision=6, linewidth=120)
        directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
        print(f"Saving files to {directory}")
        test_dtw_subseq1()
