import pytest
import numpy as np
import sys
sys.path.append(".")
from dtaidistance import dtw, dtw_c

def test_distance1_a():
    # dist_opts = {'max_dist': 0.201, 'max_step': 0.011, 'max_length_diff': 8, 'window': 3}
    dist_opts = {'window': 3}
    s1 = np.array([ 0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.])
    s2 = np.array([ 0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.])
    d1 = dtw.distance(s1, s2, **dist_opts)
    d2 = dtw_c.distance_nogil(s1, s2, **dist_opts)
    assert d1 == d2
    assert d1 == pytest.approx(0.0004)

def test_distance1_b():
    dist_opts = {}
    s1 = np.array([ 0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.])
    s2 = np.array([ 0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.])
    d1 = dtw.distance(s1, s2, **dist_opts)
    d2 = dtw_c.distance_nogil(s1, s2, **dist_opts)
    assert d1 == d2
    assert d1 == pytest.approx(0.0004)

test_distance1_a()
