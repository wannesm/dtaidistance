# -*- coding: UTF-8 -*-
"""
dtaidistance.affinedtw
~~~~~~~~~~~~~~~~~~~~~~

Affine Dynamic Time Warping (DTW)

Based on:

    T.-W. Chen, M. Abdelmaseeh, and D. Stashuk.
    Affine and regional dynamic time warping. In 2015
    IEEE International Conference on Data Mining Workshop (ICDMW), 2015.

:author: Wannes Meert
:copyright: Copyright 2025 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging

from . import util_numpy
from . import dtw
from .preprocessing import _total_least_squares
from .exceptions import NumpyException

try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
except ImportError:
    np = None


logger = logging.getLogger("be.kuleuven.dtai.distance")


def warping_path(s1, s2, include_distance=False, **kwargs):
    path, d, _ = warping_path_and_coeff(s1, s2, **kwargs)
    if include_distance:
        return path, d
    return path


def warping_path_and_coeff_fast(s1, s2, nb_steps=10, znormalization=True,
                                **kwargs):
    kwargs["use_c"] = True
    kwargs["use_pruning"] = True
    return warping_path_and_coeff(s1, s2, nb_steps=nb_steps, znormalization=znormalization, **kwargs)


def warping_path_and_coeff(s1, s2, nb_steps=10, znormalization=True,
                           **kwargs):
    """
    :param s1: From series
    :param s2: To series
    :param nb_steps:
    :param znormalization: First perform z-normalization. This is a cheap
        operation that brings values already closer together and makes
        the affine transformation more stable.
    """
    if np is None:
        raise NumpyException("warping_path_and_coeff needs numpy")
    if znormalization:
        f_mean, f_std = np.mean(s1), np.std(s1)
        t_mean, t_std = np.mean(s2), np.std(s2)
        s1 = (s1 - f_mean) / f_std
        s2 = (s2 - t_mean) / t_std
    path, d = dtw.warping_path(s1, s2, include_distance=True, **kwargs)
    s1a = s1
    d_prev = np.inf
    d_stop = 0.01*d
    a2, b2, path2, d2 = 1, 0, path, d
    for i_step in range(nb_steps):
        if i_step > 0:
            d_prev = d2
            path2, d2 = dtw.warping_path(s1a, s2, include_distance=True, **kwargs)
        if d_prev - d2 < d_stop:
            a, b, path, d = a2, b2, path2, d2
            break
        path2 = np.array(path2)
        xi, yi = path2[:,0], path2[:,1]
        x, y = s1[xi], s2[yi]
        b3, a3 = _total_least_squares(x.reshape(-1,1), y)
        if a3 < 0.01:
            # slope too flat, might flip the series, stop here
            a, b, path, d = a2, b2, path2, d2
            break
        else:
            a2, b2 = a3, b3
            s1a = a2*s1+b2
    else:
        a, b, path, d = a2, b2, path2, d2
    if znormalization:
        # z-normalization is also an affine transformation, combine both
        a = a*t_std/f_std
        b = t_std*b + t_mean - a*f_mean*t_std/f_std
    return path, d, (b, a)


def scale_fast(s1, s2, **kwargs):
    kwargs["use_c"] = True
    kwargs["use_pruning"] = True
    return scale(s1, s2, **kwargs)

def scale(s1, s2, **kwargs):
    """
    :param s1: From series
    :param s2: To series
    :param kwargs: Arguments passed on to warping_path_and coeff
    """
    _, _, (b, a) = warping_path_and_coeff(s1, s2, **kwargs)
    s1 = a*s1 + b
    return s1

