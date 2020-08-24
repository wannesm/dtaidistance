import sys
import tempfile
import logging
import pytest
from pathlib import Path
from dtaidistance import util_numpy, dtw_weighted as dtww
import dtaidistance.dtw_visualisation as dtwvis


numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")
logger = logging.getLogger("be.kuleuven.dtai.distance")


def get_directory(directory):
    if directory is not None:
        return Path(directory)
    with tempfile.TemporaryDirectory() as directory_name:
        return Path(directory_name)


@numpyonly
def test_split():
    with util_numpy.test_uses_numpy() as np:
        values = np.array([1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9])
        targets = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0])
        ig, thr, _h0 = dtww.DecisionTreeClassifier.informationgain_continuous(targets, values)
        assert thr == pytest.approx(4.5)


@numpyonly
def test_split2():
    with util_numpy.test_uses_numpy() as np:
        values = np.array([0., 0., 0.])
        targets = np.array([0, 1, 0])
        ig, thr, _h0 = dtww.DecisionTreeClassifier.informationgain_continuous(targets, values)
        assert ig == pytest.approx(0.0)
        assert thr is None


@numpyonly
def test_kdistance():
    with util_numpy.test_uses_numpy() as np:
        values = np.array([1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9])
        thr = 4.5
        kd = dtww.DecisionTreeClassifier.kdistance(values, thr)
        assert kd == pytest.approx(1.5)


@numpyonly
def test_kdistance2():
    with util_numpy.test_uses_numpy() as np:
        values = np.array([0., 0., 0.])
        thr = 0.0
        kd = dtww.DecisionTreeClassifier.kdistance(values, thr)
        assert kd == pytest.approx(0.0)


@numpyonly
def test_decisiontree(directory=None):
    with util_numpy.test_uses_numpy() as np:
        features = np.array([
            [0.5395256916996046, 0.5925000000000002],
            [0.507905138339921, 0.6900000000000002],
            [0.7430830039525692, 0.7150000000000001],
            [0.7391304347826088, 0.7300000000000002],
            [0.6857707509881423, 0.4700000000000002],
            [0.7272727272727273, 0.40500000000000014],
            [0.6936758893280632, 0.4125000000000002],
            [0.6897233201581027, 0.26000000000000023],
            [0.616600790513834, 0.5025000000000002],
            [0.5810276679841897, 0.4550000000000002],
            [0.4841897233201582, 0.3875000000000002],
            [0.3181818181818181, 0.3600000000000001],
            [0.28063241106719367, 0.47250000000000014],
            [0.2549407114624505, 0.5725000000000002],
            [0.39920948616600793, 0.6125000000000002],
            [0.39525691699604737, 0.6175000000000002],
            [0.375494071146245, 0.6475000000000001],
            [0.3359683794466403, 0.6350000000000001],
            [0.34584980237154145, 0.7275000000000001],
            [0.38537549407114624, 0.7375000000000002],
            [0.2075098814229248, 0.8650000000000001],
            [0.3774703557312252, 0.7600000000000001],
            [0.4624505928853755, 0.7500000000000001],
            [0.5276679841897233, 0.8425],
            [0.6383399209486166, 0.8925000000000001],
            [0.6798418972332015, 0.8275000000000001],
            [0.782608695652174, 0.7550000000000001],
            [0.7608695652173912, 0.5575000000000001],
            [0.8537549407114624, 0.5550000000000002],
            [0.8972332015810277, 0.27000000000000024],
            [0.7549407114624507, 0.1575000000000003],
            [0.5790513833992094, 0.1525000000000002],
            [0.5118577075098814, 0.2100000000000002],
            [0.43083003952569165, 0.03500000000000014],
            [0.4209486166007905, 0.05500000000000016],
            [0.3320158102766798, 0.16000000000000025],
            [0.22332015810276673, 0.05250000000000021],
            [0.011857707509881382, 0.2975000000000001],
            [0.14229249011857703, 0.4425000000000002],
            [0.19565217391304346, 0.5900000000000001]
        ])
        targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print('Matplotlib not installed')
                return
            plt.figure(figsize=(3, 3))
            plt.scatter(features[:20, 0], features[:20, 1], marker="+")
            plt.scatter(features[20:, 0], features[20:, 1], marker=".")
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.savefig(str(directory / "features.png"))
            plt.close()

        clf = dtww.DecisionTreeClassifier()
        clf.fit(features, targets, use_feature_once=False)

        if directory:
            try:
                from sklearn.tree import export_graphviz
            except ImportError:
                return
            export_graphviz(clf, out_file=str(directory / "hierarchy.dot"))


if __name__ == "__main__":
    # Print settings
    np.set_printoptions(precision=2)
    # Logger settings
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = 0
    # Output path
    directory = Path(__file__).parent / "output"

    # test_split()
    # test_split2()
    # test_kdistance()
    # test_kdistance2()
    test_decisiontree(directory)
