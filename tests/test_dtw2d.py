import math
import pytest
import numpy as np
from dtaidistance import dtwndim


def test_distance1_a():
    s1 = np.array([[0, 0], [0, 1], [2, 1], [0, 1],  [0, 0]], dtype=np.double)
    s2 = np.array([[0, 0], [2, 1], [0, 1], [0, .5], [0, 0]], dtype=np.double)
    d1 = dtwndim.distance(s1, s2)
    print(d1)
    d1p, paths = dtwndim.warping_paths(s1, s2)
    print(d1p)
    print(paths)
    assert d1 == pytest.approx(d1p)


if __name__ == "__main__":
    test_distance1_a()
