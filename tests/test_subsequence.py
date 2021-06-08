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
        print(sa.matching_function)
        if not dtwvis.test_without_visualization():
            import matplotlib.pyplot as plt
            if directory:
                plt.plot(sa.matching_function)
                plt.savefig(directory / "subseq_matching.png")
                plt.close()


if __name__ == "__main__":
    with util_numpy.test_uses_numpy() as np:
        np.set_printoptions(precision=6, linewidth=120)
        directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
        print(f"Saving files to {directory}")
        test_dtw_subseq1()
