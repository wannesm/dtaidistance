import logging
import sys, os
import random
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest

from dtaidistance import dtw, util_numpy
from dtaidistance import dtw_visualisation as dtwvis

directory = None
logger = logging.getLogger("be.kuleuven.dtai.distance")
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")
pandasonly = pytest.mark.skipif("util_numpy.test_without_pandas()")


@numpyonly
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


@numpyonly
def test_distance1_b():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {}
        s1 = np.array([ 0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.])
        s2 = np.array([ 0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        d3, wps = dtw.warping_paths(s1, s2, **dist_opts)
        print(np.power(wps,2))
        assert d1 == d2
        assert d1 == d3
        assert d1 == pytest.approx(0.02)


@numpyonly
def test_distance2_a():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {'max_dist': 1.1}
        s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
        s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        assert d1 == d2
        assert d1 == pytest.approx(1.0)


@numpyonly
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


@numpyonly
def test_distance2_b():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {'max_step': 1.1}
        s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
        s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        assert d1 == d2
        assert d1 == pytest.approx(1.0)


@numpyonly
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


@numpyonly
def test_distance2_c():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {}
        s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
        s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        d1 = dtw.distance(s1, s2, **dist_opts)
        d2 = dtw.distance_fast(s1, s2, **dist_opts)
        assert d1 == d2
        assert d1 == pytest.approx(1.0)


@numpyonly
def test_distance3_a():
    with util_numpy.test_uses_numpy() as np:
        dist_opts = {"penalty": 0.005, "max_step": 0.011, "window": 3}
        s = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.005, 0.01, 0.015, 0.02, 0.01, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        p = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.005, 0.01, 0.015, 0.02, 0.01, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        d1 = dtw.distance(s, p, **dist_opts)
        d2 = dtw.distance_fast(s, p, **dist_opts)
        assert d1 == pytest.approx(d2)


@pytest.mark.skip("Pandas is not a mandatory requirement")
@numpyonly
@pandasonly
def test_distance4():
    with util_numpy.test_uses_numpy(strict=False) as np, util_numpy.test_uses_pandas() as pd:
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


@numpyonly
def test_distance6():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double)
        s2 = np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0])
        d = dtw.distance_fast(s1, s2, window=2)
        # print(d)


@numpyonly
def test_bug1():
    """Failed on Windows if pointer types are different."""
    with util_numpy.test_uses_numpy() as np:
        series = [np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double),
                  np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
                  np.array([0.0, 0, 1, 2, 1, 0, 0, 0])]
        ds = dtw.distance_matrix_fast(series)
        # print(ds)


@numpyonly
def test_bug1_serial():
    """Failed on Windows if pointer types are different."""
    with util_numpy.test_uses_numpy() as np:
        series = [np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double),
                  np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
                  np.array([0.0, 0, 1, 2, 1, 0, 0, 0])]
        ds = dtw.distance_matrix_fast(series, parallel=False)
        # print(ds)


@numpyonly
def test_bug1_psi():
    with util_numpy.test_uses_numpy() as np:
        s = [np.array([0., 0, 1, 2, 1, 0, 1, 0, 0]),
             np.array([9., 0, 1, 2, 1, 0, 1, 0, 9])]
        res1 = dtw.distance_matrix(s, compact=True, psi=1)
        res2 = dtw.distance_matrix_fast(s, compact=True, psi=1)
        print(res1)
        print(res2)
        assert res1 == pytest.approx(res2)


@numpyonly
def test_bug2():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([5.005335029629605081e-01, 5.157722489130834864e-01, 4.804319657333316340e-01, 4.520537745752661318e-01, 4.867408184050183717e-01, 4.806534229629605415e-01, 4.530552579964135518e-01, 4.667067057333316171e-01, 4.567955137333316040e-01, 4.414902037333315876e-01, 4.240597964014319321e-01, 4.225263829008334970e-01, 4.030970017333316280e-01, 4.404482984865574768e-01, 3.852339312962939077e-01, 3.634947117333316435e-01, 3.861488867383516266e-01, 3.413363679008334928e-01, 3.451913457333316004e-01, 3.695692377333316680e-01, 3.434781337333315809e-01, 3.063217006568062506e-01, 2.845283817333316145e-01, 2.955394357333315791e-01, 3.151374838781335619e-01, 2.561411067352764026e-01, 2.301194263297469400e-01, 2.478605028202762184e-01, 1.972828198566299318e-01, 2.150545617333316228e-01, 2.232865857333316273e-01, 2.492665580680986370e-01, 2.144049374050155388e-01, 2.079081117333316520e-01, 1.879600957333316391e-01, 1.638555197333316227e-01, 1.425566689000865583e-01, 2.016327177333316067e-01, 2.290943870240647606e-01, 1.900932117333316296e-01, 1.503233018025057766e-01, 1.970833717333316248e-01, 1.999393777333316191e-01, 2.018818837333316019e-01, 2.554168153357214144e-01, 2.345002377333316179e-01, 2.407103957333316113e-01, 2.762874997333316096e-01, 3.059693477333316203e-01, 3.328774862341668528e-01, 3.583867537333316200e-01, 3.743879884050183016e-01, 4.266385131705089373e-01, 4.445410410742424712e-01, 4.642271795675002033e-01, 4.402678696630802357e-01, 4.814591396296271641e-01, 5.317886460815400840e-01, 5.548714817383517683e-01, 5.062713000716849709e-01, 5.431524597333317050e-01, 5.537961812962939323e-01, 5.720852595675002261e-01, 5.933977447347652534e-01, 5.845479257333316969e-01, 6.133363017333317568e-01, 6.276481431102108877e-01, 6.132085097333317414e-01, 5.922371597333316862e-01, 5.778388756463566089e-01])
        s2 = np.array([5.584292601075275808e-01, 5.214504501075275522e-01, 4.877978901075275542e-01, 5.078206201075274873e-01, 4.769738701075275644e-01, 4.478925501075275428e-01, 4.242528301075275676e-01, 4.307546401075275644e-01, 4.370594201075275187e-01, 4.331284101075275617e-01, 4.810766301075275475e-01, 4.250942801075275335e-01, 3.973955801075275684e-01, 4.380910701075275693e-01, 3.786794801075275552e-01, 3.850050201075275180e-01, 3.576176301075275621e-01, 2.987050201075275302e-01, 3.377542001075275468e-01, 3.262601401075275187e-01, 3.278248801075275276e-01, 3.347294101075275474e-01, 3.222199801075275594e-01, 3.372712101075275304e-01, 2.526810801075275448e-01, 1.774206901075275622e-01, 2.384015601075275825e-01, 2.419624201075275816e-01, 1.694136001075275677e-01, 1.983933401075275715e-01, 2.272449101075275646e-01, 1.490059201075275563e-01, 1.416013701075275744e-01, 1.997542401075275698e-01, 1.791462801075275613e-01, 1.712680901075275819e-01, 1.851759601075275707e-01, 1.450854801075275591e-01, 1.041379601075275718e-01, 9.028068310752757064e-02, 1.358144301075275839e-01, 2.006444701075275616e-01, 2.003521501075275768e-01, 2.100136501075275663e-01, 2.521797401075275280e-01, 2.364524601075275734e-01, 2.236850301075275771e-01, 2.873612101075275205e-01, 3.358473801075275156e-01, 3.288144201075275386e-01, 3.195859301075275605e-01, 3.482947201075275445e-01, 4.032929801075275655e-01, 4.566962501075275682e-01, 5.173766201075274962e-01, 5.463256501075275384e-01, 5.172673701075275465e-01, 5.054312901075275200e-01, 5.344046101075274890e-01, 5.389180101075274898e-01, 5.188896901075275014e-01, 5.484243401075274971e-01, 5.899157901075275934e-01, 5.987863201075275255e-01, 6.357147701075275270e-01, 6.277379101075275525e-01, 5.519873201075274904e-01, 5.634240801075275362e-01, 6.307956401075275332e-01, 6.488636001075275272e-01])
        res1 = dtw.distance(s1, s2)
        res2 = dtw.distance(s1, s2, max_dist=.20)
        res3, _m3 = dtw.warping_paths(s1, s2)
        res4, _m4 = dtw.warping_paths(s1, s2, max_dist=.20)
        # print(res1)
        # print(res2)
        # print(res3)
        # print(res4)
        # np.savetxt('/Users/wannes/Desktop/debug/m3.txt', m3)
        # np.savetxt('/Users/wannes/Desktop/debug/m4.txt', m4)
        assert res1 == pytest.approx(res2)
        assert res1 == pytest.approx(res3)
        assert res1 == pytest.approx(res4)


# @numpyonly
# def test_bug3():
#     with util_numpy.test_uses_numpy() as np:
#         psi = 30
#
#         x = np.arange(0, 20, .5)
#         s1 = np.sin(x)
#         s2 = np.sin(x - 1)
#         random.seed(1)
#         for idx in range(len(s2)):
#             if random.random() < 0.05:
#                 s2[idx] += (random.random() - 0.5) / 2
#         d, paths = dtw.warping_paths(s1, s2, window=25, psi=psi)
#         best_path = dtw.best_path(paths)
#
#         with util_numpy.test_uses_numpy() as np:
#             np.set_printoptions(precision=3, linewidth=300, threshold=np.inf)
#             print(paths)
#
#         if directory and not dtwvis.test_without_visualization():
#             dtwvis.plot_warping(s1, s2, best_path, filename=directory / 'bug3_warping.png')
#             dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename=directory / 'bug3_warpingpaths.png')


@numpyonly
def test_bug4():
    with util_numpy.test_uses_numpy() as np:
        psi = 1
        win = 10

        x = np.arange(0, 13, .5)
        s1 = np.sin(x)
        s2 = np.sin(x - 1)

        random.seed(1)
        for idx in range(len(s2)):
            if random.random() < 0.05:
                s2[idx] += (random.random() - 0.5) / 2

        d1, paths = dtw.warping_paths(s1, s2, window=win, psi=psi, psi_neg=False)
        best_path = dtw.best_path(paths)

        d2 = dtw.distance_fast(s1, s2, psi=psi, window=win)
        d3 = dtw.distance(s1, s2, psi=psi, window=win)

        # print(f'{d1=}, {d2=}, {d3=}')
        # with util_numpy.test_uses_numpy() as np:
        #     np.set_printoptions(precision=3, linewidth=30000, threshold=np.inf)
        #     # print(np.power(paths, 2))
        #     print(paths)

        if directory and not dtwvis.test_without_visualization():
            dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename=directory / 'bug4_warpingpaths.png')

        assert d1 == pytest.approx(0.6305018693852942), 'dtw.warping_paths failed'
        assert d2 == pytest.approx(0.6305018693852942), 'dtw.distance_fast failed'
        assert d3 == pytest.approx(0.6305018693852942), 'dtw.distance failed'


@numpyonly
def test_bug_size():
    """Two series of length 1500 should not trigger a size error.

    The warping paths matrix is of size 1501**2 = 2_253_001.
    If using 64bit values: 1501**2*64/(8*1024*1024) = 17.2MiB.
    """
    with util_numpy.test_uses_numpy() as np:
        s1 = np.random.rand(1500)
        s2 = np.random.rand(1500)
        d1, _ = dtw.warping_paths_fast(s1, s2)
        d2, _ = dtw.warping_paths(s1, s2)
        assert d1 == pytest.approx(d2)


@numpyonly
def test_bug5_path():
    """
    without psi: [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4)]
    with psi:            [(0, 1), (1, 2), (1, 3), (2, 4)]

    Why is this not (with psi): [(2,4), (1,3), (0,2)] ?
    Answer:
    Numerical inaccuracies. When choosing the best path from (1,3) the
    three options are [1.0, 1.9999999999999996, 0.9999999999999996].
    Thus moving left (last one) is chosen instead of the expected diagonal.

    In theory:
    Path 1: (0,2), (1,3), (2,4) = sqrt(1**2 + 0 + 0) = 1
    Path 2: (0,1), (1,2), (1,3), (2,4) = sqrt(0 + 1**2 + 0 + 0) = 1
    And path 1 should be chosen because the diagonal move has priority.

    In practice, floating point inaccuracies:
    Path 1: (2.1-3.1) = 1.0
    Path 2: (4.1-3.1) = 0.9999999999999996

    """
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([2.1, 4.1, 5.1])
        s2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        d1, wps = dtw.warping_paths(s1, s2, psi=[0, 0, len(s2), len(s2)])
        best_path = dtw.best_path(wps)
        print(best_path)

        # if directory and not dtwvis.test_without_visualization():
        #     dtwvis.plot_warpingpaths(s1, s2, wps, best_path, filename=directory / 'bug5_warpingpaths.png')
        #     dtwvis.plot_matrix(wps, shownumbers=True, filename=directory / 'bug5_matrix.png')


@numpyonly
def test_bug6():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0.0, 1.0])
        s2 = np.array([0.0, 0.0])

        psi = [0, 1, 0, 0]
        d, paths = dtw.warping_paths(s1, s2, psi=psi, use_c=True)
        assert d == pytest.approx(0.0)
        d = dtw.distance(s1, s2, psi=psi, use_c=True)
        assert d == pytest.approx(0.0)
        d = dtw.distance(s1, s2, psi=psi, use_c=False)
        assert d == pytest.approx(0.0)

        psi = [0, 0, 0, 1]
        d, paths = dtw.warping_paths(s1, s2, psi=psi, use_c=True)
        assert d == pytest.approx(1.0)
        d = dtw.distance(s1, s2, psi=psi, use_c=True)
        assert d == pytest.approx(1.0)
        d = dtw.distance(s1, s2, psi=psi, use_c=False)
        assert d == pytest.approx(1.0)


if __name__ == "__main__":
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    # with util_numpy.test_uses_numpy() as np:
    #     np.set_printoptions(precision=2, linewidth=120)
    logger.setLevel(logging.WARNING)
    sh = logging.StreamHandler(sys.stdout)
    logger.addHandler(sh)
    test_bug1()
    # test_distance1_a()
    # test_distance1_b()
    # test_distance2_a()
    # test_distance2_b()
    # test_distance2_c()
    # test_distance3_a()
    # test_distance4()
    # test_distance6()
    # test_bug1_psi()
    # test_bug2()
    # test_bug3()
    # test_bug5_path()
    # test_bug_size()
