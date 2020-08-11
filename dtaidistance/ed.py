# -*- coding: UTF-8 -*-
"""
dtaidistance.ed
~~~~~~~~~~~~~~~

Euclidean Distance (ED)

:author: Wannes Meert
:copyright: Copyright 2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import math

from . import util_numpy


logger = logging.getLogger("be.kuleuven.dtai.distance")


ed_cc = None
try:
    from . import ed_cc
except ImportError:
    logger.debug('DTAIDistance C library not available')
    ed_cc = None


try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
except ImportError:
    np = None


def _check_library(raise_exception=True):
    if ed_cc is None:
        msg = "The compiled dtaidistance C library is not available.\n" + \
              "See the documentation for alternative installation options."
        logger.error(msg)
        if raise_exception:
            raise Exception(msg)


def distance(s1, s2):
    """ Euclidean distance between two sequences. Supports different lengths.

    If the two series differ in length, compare the last element of the shortest series
    to the remaining elements in the longer series. This is compatible with Euclidean
    distance being used as an upper bound for DTW.

    :param s1: Sequence of numbers
    :param s2: Sequence of numbers
    :return: Euclidean distance
    """
    n = min(len(s1), len(s2))
    ub = 0
    for v1, v2 in zip(s1, s2):
        ub += (v1 - v2)**2
    # If the two series differ in length, compare the last element of the shortest series
    # to the remaining elements in the longer series
    if len(s1) > len(s2):
        v2 = s2[n - 1]
        for v1 in s1[n:]:
            ub += (v1 - v2)**2
    elif len(s1) < len(s2):
        v1 = s1[n-1]
        for v2 in s2[n:]:
            ub += (v1 - v2)**2
    return math.sqrt(ub)


def distance_fast(s1, s2):
    _check_library(raise_exception=True)
    # Check that Numpy arrays for C contiguous
    s1 = util_numpy.verify_np_array(s1)
    s2 = util_numpy.verify_np_array(s2)
    # Move data to C library
    d = ed_cc.distance(s1, s2)
    return d
