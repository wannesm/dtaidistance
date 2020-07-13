import logging
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest

from dtaidistance import dtw, util_numpy


logger = logging.getLogger("be.kuleuven.dtai.distance")


def test_distance1_a():
    with util_numpy.test_uses_numpy() as np:
        # dist_opts = {'max_dist': 0.201, 'max_step': 0.011, 'max_length_diff': 8, 'window': 3}
        dist_opts = {'window': 3}
        s1 = np.array([ 0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.])
        s2 = np.array([ 0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        print("X")
        assert d1 == d2
        assert d1 == pytest.approx(0.02)


def test_distance1_b():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {}
        s1 = np.array([ 0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.])
        s2 = np.array([ 0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        assert d1 == d2
        assert d1 == pytest.approx(0.02)


def test_distance2_a():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {'max_dist': 1.1}
        s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
        s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        assert d1 == d2
        assert d1 == pytest.approx(1.0)


def test_distance2_aa():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {'max_dist': 0.1}
        s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
        s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        print(d1, d2)
        assert d1 == d2
        assert d1 == pytest.approx(np.inf)


def test_distance2_b():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {'max_step': 1.1}
        s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
        s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        assert d1 == d2
        assert d1 == pytest.approx(1.0)


def test_distance2_bb():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {'max_step': 0.1}
        s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
        s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        print(d1, d2)
        assert d1 == d2
        assert d1 == pytest.approx(np.inf)


def test_distance2_c():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {}
        s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
        s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        assert d1 == d2
        assert d1 == pytest.approx(1.0)


def test_distance3_a():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {"penalty": 0.005, "max_step": 0.011, "window": 3}
        s = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.005, 0.01, 0.015, 0.02, 0.01, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        p = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.005, 0.01, 0.015, 0.02, 0.01, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        d1 = dtw.distance(s, p, **dist_opts)
        d2 = dtw.distance_fast(s, p, **dist_opts)
        assert d1 == pytest.approx(d2)


def test_distance4():
    with util_numpy.test_uses_numpy(strict=False) as np:
        try:
            import pandas as pd
        except ImportError:
            # If no pandas, ignore test (not a required dependency)
            return
        s = [[0.,    0.,   0.,    0.,   0.,   0., 0., 0., 0., 0., 0., 0., 0.],
             [0.005, 0.01, 0.015, 0.02, 0.01, 0., 0., 0., 0., 0., 0., 0., 0.],
             [0.,    0.,   0.,    0.,   0.,   0., 0., 0., 0., 0., 0., 0., 0.]]
        p = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        df = pd.DataFrame(data=s)
        s = df.values
        for i in range(s.shape[0]):
            ss = s[i]  # ss will not be C contiguous memory layout
            d = dtw.distance_fast(ss, p)
            # print(d)


# def test_distance5():
#     healthy = np.array([-0.01014404, 0.01240234, -0.00549316, 0.01905518, 0.02086182,
#                      0.02155762, 0.02252197, -0.0015625, 0.0194458, -0.00305176,
#                      0.01724854, 0.01274414, 0.01470947, 0.01373291, -0.00751953,
#                      0.01088867, -0.01018066, 0.01325684, 0.00531006, 0.01184082,
#                      0.01030273, -0.00766602, 0.00996094, -0.01044922, 0.00991211,
#                      0.00155029, 0.01335449, 0.0135498, -0.00367432, 0.00953369,
#                      -0.01192627, 0.01107178, -0.00112305, 0.01309814, 0.01253662,
#                      -0.00327148, 0.00714111, -0.01375732, 0.00942383, -0.00631104,
#                      0.015271, 0.01461182, 0.00447998, 0.01408691, -0.00461426,
#                      0.01923828, -0.00228271, 0.01993408, 0.0177124, 0.01256104])
#
#     faulty = np.array([0.51872559, 0.51743164, 0.51727295, 0.51866455, 0.512146,
#                     0.5309082, 0.52078857, 0.52185059, 0.52429199, 0.52486572,
#                     0.53078613, 0.50743408, 0.52678223, 0.52731934, 0.52879639,
#                     0.53051758, 0.51055908, 0.54437256, 0.5453125, 0.54205322,
#                     0.54060059, 0.53500977, 0.54443359, 0.52835693, 0.53216553,
#                     0.53133545, 0.53546143, 0.53426514, 0.50535889, 0.53413086,
#                     0.53583984, 0.53778076, 0.53405762, 0.51973877, 0.54488525,
#                     0.53464355, 0.5338501, 0.53098145, 0.528479, 0.53360596,
#                     0.50834961, 0.52283936, 0.52408447, 0.53001709, 0.5282959,
#                     0.50821533, 0.5369873, 0.53790283, 0.53980713, 0.53851318])
    # d = dtw.distance_fast(healthy, faulty)
    # print(d)
    # dp, dsp = ssdtw.warping_paths(healthy, faulty, None)
    # print(dp)
    # d, ds = dtw.warping_paths(healthy, faulty)
    # print(d)
    # print(ds)
    # dtw.plot(healthy, faulty, ds, "/Users/wannes/Desktop/test1.png")
    # dtw.plot(healthy, faulty, dsp, "/Users/wannes/Desktop/test2.png")
    # np.savetxt("/Users/wannes/Desktop/matrix1.txt", ds)
    # np.savetxt("/Users/wannes/Desktop/matrix2.txt", dsp)
    # print('healthy', np.mean(healthy), np.std(healthy))
    # print('faulty', np.mean(faulty), np.std(faulty))
    #
    # Conclusion: Constant difference between two series will always have the diagonal as best solution and is thus
    #             equal to Euclidean distance. This is one of the reasons why normalisation is important for
    #             clustering.


def test_distance6():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double)
        s2 = np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0])
        d = dtw.distance_fast(s1, s2, window=2)
        # print(d)


def test_bug1():
    """Failed on Windows if pointer types are different."""
    with util_numpy.test_uses_numpy() as np:
        series = [np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double),
                  np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
                  np.array([0.0, 0, 1, 2, 1, 0, 0, 0])]
        ds = dtw.distance_matrix_fast(series)
        # print(ds)


def test_bug1_serial():
    """Failed on Windows if pointer types are different."""
    with util_numpy.test_uses_numpy() as np:
        series = [np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double),
                  np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
                  np.array([0.0, 0, 1, 2, 1, 0, 0, 0])]
        ds = dtw.distance_matrix_fast(series, parallel=False)
        # print(ds)


def test_bug1_psi():
    with util_numpy.test_uses_numpy() as np:
        s = [np.array([0., 0, 1, 2, 1, 0, 1, 0, 0]),
             np.array([9., 0, 1, 2, 1, 0, 1, 0, 9])]
        res1 = dtw.distance_matrix(s, compact=True, psi=1)
        res2 = dtw.distance_matrix_fast(s, compact=True, psi=1)
        print(res1)
        print(res2)
        assert res1 == pytest.approx(res2)


if __name__ == "__main__":
    logger.setLevel(logging.WARNING)
    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(sh)
    # test_bug1()
    test_distance1_a()
    # test_distance2_a()
    # test_distance2_b()
    # test_distance2_c()
    # test_distance3_a()
    # test_distance4()
    # test_distance6()
    # test_bug1_psi()
