import math
import array

import pytest

from dtaidistance import dtw, dtw_cc, util_numpy


numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


def test_distance1_array():
    s1 = array.array('d', [0, 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = array.array('d', [0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw_cc.distance(s1, s2)
    # print(f'd = {d}')
    assert d == pytest.approx(math.sqrt(2))


@numpyonly
def test_distance1_numpy():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0])
        s2 = np.array([0., 1, 2, 0, 0, 0, 0, 0, 0])
        d = dtw.distance_fast(s1, s2)
        # print(f'd = {d}')
        assert d == pytest.approx(math.sqrt(2))


if __name__ == "__main__":
    test_distance1_array()
    # test_distance1_numpy(,)
