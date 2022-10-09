import math
import pytest
import time
from dtaidistance import ed, util_numpy, dtw
from dtaidistance.subsequence.dtw import subsequence_search


numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_distance1_a():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.float32)
        s2 = np.array([0., 1, 2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        print(f's1 itemsize = {s1.itemsize}')

        d = dtw.distance_fast(s1, s2)
        # print('d = {}'.format(d))
        assert (d) == pytest.approx(1.4142135381698608)

        d = ed.distance_fast(s1, s2)
        # print('d = {}'.format(d))
        assert(d) == pytest.approx(2.8284271247461903)


@numpyonly
@pytest.mark.benchmark(group="subseqsearch_eeg")
def test_dtw_subseqsearch_eeg():
    with util_numpy.test_uses_numpy() as np:
        from test_subsequence import create_data_subseqsearch_eeg
        query, s, k, series, s_idx = create_data_subseqsearch_eeg(np, dtype=np.float32)
        tic = time.perf_counter()
        sa = subsequence_search(query, s, dists_options={'use_c': True})
        best = sa.kbest_matches_fast(k=k)
        toc = time.perf_counter()
        print("Searching performed in {:0.4f} seconds".format(toc - tic))
        # print(sa.distances)
        # print(best)


if __name__ == "__main__":
    test_distance1_a()
    # test_dtw_subseqsearch_eeg()
