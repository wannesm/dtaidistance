import math
import pytest
import numpy as np
from dtaidistance import dtw

np.set_printoptions(precision=2, linewidth=120)


def test_penalty_cyclicalshift():
    s1 = np.array([0., 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0])
    s2 = np.array([2., 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2])
    # plt.plot(s1)
    # plt.plot(s2)
    # plt.show(block=True)
    d1 = dtw.distance(s1, s2)
    d2 = dtw.distance(s1, s2, penalty=1)
    assert d1 == pytest.approx(math.sqrt(10))
    assert d2 == pytest.approx(math.sqrt(14))


if __name__ == "__main__":
    test_penalty_cyclicalshift()
