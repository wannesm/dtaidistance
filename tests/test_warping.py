import math
import pytest
import numpy as np
from dtaidistance import dtw, dtw_c


def test_normalize():
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
    s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
    r = dtw.warp(s1, s2, plot=False)
    r_c = np.array([0., 1., 2., 2., 1., 0.25, 0., 0., 2., 1., 0., 0., 0.])
    # dtw.plot_warping(s1, s2)
    np.testing.assert_almost_equal(r, r_c, decimal=4)


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=120)
    test_normalize()
