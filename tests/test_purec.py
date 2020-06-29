import math
import array
import pytest
from dtaidistance import dtw, dtw_cc


def test_distance1_array():
    import array
    s1 = array.array('d', [0, 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = array.array('d', [0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw_cc.distance(s1, s2)
    print(f'd = {d}')
    assert(d) == pytest.approx(math.sqrt(2))


def test_distance1_numpy():
    try:
        import numpy as np
    except ImportError:
        # Optional test if numpy is present
        return
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = np.array([0., 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2)
    print(f'd = {d}')
    assert(d) == pytest.approx(math.sqrt(2))


if __name__ == "__main__":
    test_distance1_array()
    # test_distance1_numpy(,)
