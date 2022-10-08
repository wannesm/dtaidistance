import math
import pytest
from dtaidistance import ed, util_numpy, dtw


numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_distance1_a():
    with util_numpy.test_uses_numpy() as np:
        s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.float32)
        s2 = np.array([0., 1, 2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        print(f's1 itemsize = {s1.itemsize}')

        d = dtw.distance_fast(s1, s2)
        print('d = {}'.format(d))

        d = ed.distance_fast(s1, s2)
        print('d = {}'.format(d))
        assert(d) == pytest.approx(2.8284271247461903)


if __name__ == "__main__":
    test_distance1_a()
