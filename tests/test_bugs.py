import pytest
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from dtaidistance import dtw, dtw_c

if dtw_c is None:
    print('ERROR: dtw_c is not build')
    sys.exit(1)

def test_distance1_a():
    # dist_opts = {'max_dist': 0.201, 'max_step': 0.011, 'max_length_diff': 8, 'window': 3}
    dist_opts = {'window': 3}
    s1 = np.array([ 0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.])
    s2 = np.array([ 0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.])
    d1 = dtw.distance(s1, s2, **dist_opts)
    d2 = dtw_c.distance_nogil(s1, s2, **dist_opts)
    assert d1 == d2
    assert d1 == pytest.approx(0.02)


def test_distance1_b():
    dist_opts = {}
    s1 = np.array([ 0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.])
    s2 = np.array([ 0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.])
    d1 = dtw.distance(s1, s2, **dist_opts)
    d2 = dtw_c.distance_nogil(s1, s2, **dist_opts)
    assert d1 == d2
    assert d1 == pytest.approx(0.02)


def test_distance2_a():
    dist_opts = {'max_dist': 1.1}
    s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    d1 = dtw.distance(s1, s2, **dist_opts)
    d2 = dtw_c.distance_nogil(s1, s2, **dist_opts)
    print(d1, d2)
    assert d1 == d2
    assert d1 == pytest.approx(1.0)


def test_distance2_aa():
    dist_opts = {'max_dist': 0.1}
    s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    d1 = dtw.distance(s1, s2, **dist_opts)
    d2 = dtw_c.distance_nogil(s1, s2, **dist_opts)
    print(d1, d2)
    assert d1 == d2
    assert d1 == pytest.approx(np.inf)


def test_distance2_b():
    dist_opts = {'max_step': 1.1}
    s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    d1 = dtw.distance(s1, s2, **dist_opts)
    d2 = dtw_c.distance_nogil(s1, s2, **dist_opts)
    print(d1, d2)
    assert d1 == d2
    assert d1 == pytest.approx(1.0)


def test_distance2_bb():
    dist_opts = {'max_step': 0.1}
    s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    d1 = dtw.distance(s1, s2, **dist_opts)
    d2 = dtw_c.distance_nogil(s1, s2, **dist_opts)
    print(d1, d2)
    assert d1 == d2
    assert d1 == pytest.approx(np.inf)


def test_distance2_c():
    dist_opts = {}
    s1 = np.array([0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    s2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    d1 = dtw.distance(s1, s2, **dist_opts)
    d2 = dtw_c.distance_nogil(s1, s2, **dist_opts)
    assert d1 == d2
    assert d1 == pytest.approx(1.0)


def test_distance3_a():
    dist_opts = {"penalty": 0.005, "max_step": 0.011, "window": 3}
    s = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.005, 0.01, 0.015, 0.02, 0.01, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    p = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.005, 0.01, 0.015, 0.02, 0.01, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    d1 = dtw.distance(s, p, **dist_opts)
    d2 = dtw_c.distance_nogil(s, p, **dist_opts)
    assert d1 == pytest.approx(d2)


# test_distance2_a()
# test_distance2_b()
# test_distance2_c()
test_distance3_a()
