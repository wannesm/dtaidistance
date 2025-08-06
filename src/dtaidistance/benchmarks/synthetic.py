import numpy as np
import scipy as sp


def pattern1(x, x0=4, c=0.5, a=0, x1=22, d=1, r=0.0, rs=3980, w0=1, x2=25, w2=1):
    """A time series generator that simulates simple patterns in transient
    systems. It is a rising function with an overshoot and a short, one
    cycle of sine-like behavior.

    :param x0: Location of overshoot peak
    :param w0: Width of overshoot peak
    :param c: Height of converged level
    :param a: Add to height of overshoot peak
    :param x1: Location of sine wave
    :param d: Spread of sine wave
    :param r: Random noise level
    :param rs: Random seed
    """

    # Overshoot
    y = (sp.special.dawsn((x - x0)*w0) + c) * np.heaviside(x - x0, 0)
    y += (np.exp(x*w0) / (np.exp(x0*w0) / c)) * np.heaviside(x0 - x, 0)
    if a > 0:
        y += sp.stats.norm.pdf(x - x0) * a
    # Sine wave
    xd = 3
    y0 = sp.special.dawsn(xd)
    d = sp.special.dawsn((x - x1) * d)
    idx1 = d > y0
    idx2 = d < -y0
    d[idx1] = d[idx1] - y0
    d[idx2] = d[idx2] + y0
    d[~idx1 & ~idx2] = 0
    y += np.heaviside(x - x1 + xd, 0) * np.heaviside(x1 - x + xd, 0) * d / 2
    # Dip
    rv = sp.stats.norm(loc=x2, scale=w2/10)
    peak = rv.pdf(x2)
    y -= rv.pdf(x)/peak*c
    # Random noise
    if r > 0:
        np.random.seed(rs)
        y += np.random.random(y.shape) * r
    return y


def pattern2(length_of_ts, starting_index_of_wave, length_of_wave):
    return np.concatenate((np.zeros(starting_index_of_wave),
                           np.sin(np.linspace(0, 2 * np.pi, length_of_wave)),
                           np.zeros(length_of_ts - starting_index_of_wave - length_of_wave)))


def ts_under_pattern1(params_of_cluster=None, params_of_new_times=None):
    # from dtaidistance.benchmarks.synthetic import pattern1
    ys = []
    ys_new = []
    x = np.linspace(0, 30, num=200)
    x00 = 7
    x1 = 22
    if params_of_cluster == None:
        params_of_cluster = [
            (0, 0.5, 1., 1),
            (-0.7, 0.5, 1., 1),
            (-0.3, 0.5, 2., 2),
            (0.7, 0.5, 1., 2),
            (0.4, 0.5, 1., 3)
        ]

    if params_of_new_times == None:
        params_of_new_times = [
            (5, 0.5, 1, 1)
        ]

    for x0d, c, a, d in params_of_cluster:
        y = pattern1(x, x00 + x0d, c, a, x1, d)
        ys.append(y)

    for x0d, c, a, d in params_of_new_times:
        y = pattern1(x, x00 + x0d, c, a, x1, d)
        ys_new.append(y)

    return x, ys, ys_new

