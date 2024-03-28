import logging
import sys, os
import random
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest

from dtaidistance.preprocessing import differencing
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


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    test_differencing()
