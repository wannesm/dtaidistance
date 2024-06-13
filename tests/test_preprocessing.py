import logging
import sys, os
import random
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest

from dtaidistance.preprocessing import differencing, derivative
from dtaidistance import util_numpy

logger = logging.getLogger("be.kuleuven.dtai.distance")
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")
scipyonly = pytest.mark.skipif("util_numpy.test_without_scipy()")


@numpyonly
@scipyonly
def test_differencing():
    with util_numpy.test_uses_numpy() as np:
        series = np.array([0.1, 0.3, 0.2, 0.1] * 3)
        series = differencing(series, smooth=0.1)
        np.testing.assert_array_almost_equal(
            series, np.array([0.02217,  0.010307,  0.002632,  0.001504,  0.001629, -0.000457,
                              -0.001698, -0.001238, -0.004681, -0.014869, -0.026607]))

        series = np.array([[0.1, 0.3, 0.2, 0.1] * 3])
        series = differencing(series, smooth=0.1)
        np.testing.assert_array_almost_equal(
            series, np.array([[0.02217,  0.010307,  0.002632,  0.001504,  0.001629, -0.000457,
                              -0.001698, -0.001238, -0.004681, -0.014869, -0.026607]]))


@numpyonly
@scipyonly
def test_derivative():
    with util_numpy.test_uses_numpy() as np:
        def der2(qim, qip):
            return (qip-qim)

        def der3(qim, qi, qip):
            return ((qi-qim) + ((qip-qim)/2))/2

        series = np.array([0.1, 0.15, 0.27, 0.3, 0.26, 0.24, 0.2, 0.1])
        seriesds = [der2(*series[0:2])] + \
                   [der3(qim, qi, qip) for qim, qi, qip in zip(series[:-2], series[1:-1], series[2:])] + \
                   [der2(*series[-2:])]
        seriesd = derivative(series, smooth=None)
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(3, 1)
        # axs[0].plot(series)
        # for i, d in enumerate(seriesd):
        #     x = [i-1, i+1]
        #     y = [series[i]-d, series[i]+d]
        #     axs[0].plot(x, y, color="green", alpha=0.5)
        #     d = seriesds[i]
        #     y = [series[i] - d, series[i] + d]
        #     axs[0].plot(x, y, color="orange", alpha=0.5)
        # axs[1].plot(seriesd)
        # axs[2].plot(differencing(series))
        # plt.show()
        np.testing.assert_array_almost_equal(seriesd[1:-1], np.array([0.0675, 0.0975, 0.0125, -0.035, -0.025, -0.055]))
        np.testing.assert_array_almost_equal(seriesd, seriesds)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    test_differencing()
