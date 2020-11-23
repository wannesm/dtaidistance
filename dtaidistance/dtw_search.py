# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw_search
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW) Nearest Neighbor Search

This module implementing the LB_Keogh algorithm as a lower bounding measure to
improve the DTW distance function for nearest neighbor search classification
in time series data.

:author: Pieter Robberechts, Wannes Meert, Kenneth Devloo
:copyright: Copyright 2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import itertools
import array
import math
import logging

from . import util_numpy
from .dtw import distance
from .util import SeriesContainer

logger = logging.getLogger("be.kuleuven.dtai.distance")

dtw_search_cc = None
try:
    from . import dtw_search_cc
except ImportError:
    logger.debug('DTAIDistance C library not available')
    dtw_search_cc = None


try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
    list_types = (list, array.array, set, tuple, np.ndarray)
    array_min = np.min
    array_max = np.max
except ImportError:
    np = None
    list_types = (list, array.array, set, tuple)
    array_min = min
    array_max = max



def _check_library(raise_exception=True):
    if dtw_search_cc is None:
        msg = "The compiled dtaidistance C library is not available.\n" + \
              "See the documentation for alternative installation options."
        logger.error(msg)
        if raise_exception:
            raise Exception(msg)


def __lb_keogh_envelope(s, window, use_c):
        """Compute envelope of a single series."""
        sz = len(s)
        window = sz if window is None else window
        l = array.array('d', [0]*sz)
        u = array.array('d', [0]*sz)

        if use_c:
            dtw_search_cc.lb_keogh_envelope(s, l, u, window=window)
        else:
            for i in range(sz):
                imin = max(0, i - window)
                imax = min(i + window + 1, sz)
                l[i] = array_min(s[imin:imax])
                u[i] = array_max(s[imin:imax])
        return l, u

def lb_keogh_envelope_fast(data, window=None):
    """Compute time-series envelope as required by LB_Keogh.

    .. seealso:: :method:`lb_keogh_envelope`
    """
    return lb_keogh_envelope(data, window, True, True)


def lb_keogh_envelope(s, window=None, parallel=False, use_c=False, use_mp=False):
    """Compute time-series envelope as required by LB_Keogh.

    :param s: A series or iterable of series for which the envelope should be computed. 
    :param window: window to be used for the envelope generation (the envelope at time
        index i will be generated based on all observations from the time
        series at indices comprised between i-window and i+window). If None, the 
        observations from the entire series are used.
    :param parallel: Use parallel operations
    :param use_c: Use c compiled Python functions
    :param use_mp: Use Multiprocessing for parallel operations (not OpenMP)
    :returns: Lower-side of the envelope of each series in s.
    :returns: Upper-side of the envelope of each series in s.

    .. seealso:: :method:`lb_keogh`: Compute LB_Keogh similarity

    :References:

    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """

    # Check whether multiprocessing is available
    if use_c:
        _check_library(raise_exception=True)
    if parallel and (use_mp or not use_c):
        try:
            import multiprocessing as mp
            logger.info('Using multiprocessing')
        except ImportError:
            msg = 'Cannot load multiprocessing'
            logger.error(msg)
            raise Exception(msg)
    else:
        mp = None

    # If s is not an iterable
    if not isinstance(s[0], list_types): 
        logger.info('Computing envelope of s')
        return __lb_keogh_envelope(s, window, use_c)

    logger.info('Computing envelope of each series in s')
    s = SeriesContainer.wrap(s)
    d = len(s)

    lang = 'C' if use_c else 'Python'
    if use_c and parallel and not use_mp:
        logger.info("Compute envelopes in C (parallel=OMP)")
        envelopes = dtw_search_cc.lb_keogh_envelope_parallel(s, window=window)
    elif parallel and use_mp:
        logger.info("Compute envelope in {} (parallel=MP)".format(lang))
        with mp.Pool() as p:
            envelopes = p.map(__lb_keogh_envelope, [(s[di], use_c, {'window' : window}) for di in range(d)])
    else:
        logger.info("Compute envelopes in {} (parallel=No)".format(lang))
        envelopes = []
        for di in range(d):
            l, u = __lb_keogh_envelope(s[di], window, use_c)
            envelopes.append((l, u))

    return envelopes



def __lb_keogh_equal(s1, s2_envelope, use_c):
    """Compute LB Keogh of a single series."""
    if use_c:
        return dtw_search_cc.lb_keogh_from_envelope(s1, s2_envelope)
    lb = 0
    L, U = s2_envelope
    for i in range(len(s1)):
        ci = s1[i]
        dif = 0
        if ci > U[i]:
            dif = ci - U[i]
        elif ci < L[i]:
            dif = - ci + L[i]
        lb += dif
    return lb

def __lb_keogh_unequal(s1, s2, window, use_c):
    """Compute LB Keogh of a single series."""
    l1 = len(s1)
    l2 = len(s2)
    window = max(l1, l2) if window is None else window
    if use_c:
        return dtw_search_cc.lb_keogh(s1, s2, window=window)

    lb = 0
    ldiff12 = l1 + 1;
    if (ldiff12 > l2): 
        ldiff12 -= l2;
        if (ldiff12 > window):
            ldiff12 -= window
        else: 
            ldiff12 = 0
    else:
        ldiff12 = 0
    ldiff21 = l2 + window
    if (ldiff21 > l1):
        ldiff21 -= l1
    else:
        ldiff21 = 0

    for i in range(l1):
        if (i > ldiff12):
            imin = i - ldiff12
        else:
            imin = 0
        imax = max(l2, ldiff21)
        ui = array_max(s2[imin:imax])
        li = array_min(s2[imin:imax])
        ci = s1[i]
        if (ci > ui):
            lb += ci - ui
        elif (ci < li):
            lb += li - ci
    return lb

def lb_keogh_fast(s1, s2=None, s2_envelope=None, window=None):
    """Compute LB_Keogh distance.

    .. seealso:: :method:`lb_keogh`
    """
    return lb_keogh(s1, s2, s2_envelop, window, True, True)


def lb_keogh(s1, s2=None, s2_envelope=None, window=None, parallel=False, use_c=False, use_mp=False):
    """Compute LB_Keogh.

    :param s1: A series or iterable of series to compare to the envelope(s) of s2.
    :param s2: A series or iterable of candidate series. None means the
        envelope is provided via `s2_envelope` parameter and hence does not
        need to be computed again.
    :param s2_envelope: Pre-computed envelope(s) of the candidate time series.
        If set to None, it is computed based on `s2`.
    :param window: Window to be used for the envelope generation (the envelope
        at time index i will be generated based on all observations from the
        candidate series at indices comprised between i-window and i+window).
        Not used if `s2` is None.
    :param parallel: Use parallel operations
    :param use_c: Use c compiled Python functions
    :param use_mp: Use Multiprocessing for parallel operations (not OpenMP)
    :returns: Distance between the series in s1 and the envelope of the
        candidate series in s2.

    .. seealso:: :method:`lb_envelope`: Compute LB_Keogh-related envelope

    :References:

    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """
    # Check whether multiprocessing is available
    if use_c:
        _check_library(raise_exception=True)
    if parallel and (use_mp or not use_c):
        try:
            import multiprocessing as mp
            logger.info('Using multiprocessing')
        except ImportError:
            msg = 'Cannot load multiprocessing'
            logger.error(msg)
            raise Exception(msg)
    else:
        mp = None

    # Check whether all series are of equal length
    if isinstance(s1[0], list_types):
        l1 = [len(s) for s in s1]
    else:
        l1 = [len(s1)]
    if s2 is not None and isinstance(s2[0], list_types):
        l2 = [len(s) for s in s2]
    elif s2 is not None:
        l2 = [len(s2)]
    elif s2_envelope is not None and isinstance(s2_envelope[0][0], list_types):
        l2 = [len(l) for (l, _) in s2_envelope]
    else:
        l2 = [len(s2_envelope[0])]
    equal_series = all(l == l1[0] for l in l1 + l2)

    # Compute the envelopes
    if s2_envelope is None and equal_series:
        s2_envelope = lb_keogh_envelope(s2, window, parallel, use_c)
    elif s2_envelope is not None and not equal_series:
        raise ValueError("If you provide an envelope, all series should be equal in lenght.")

    # Both s1 and s2(_envelope) are a single series
    if (not isinstance(s1[0], list_types) and s2 is not None and not isinstance(s2[0], list_types)
            or s2_envelope and len(s2_envelope) == 2
            and not isinstance(s2_envelope[0][0], list_types)):
        if equal_series:
            logger.info('Computing LB Keogh distance between two series of equal lenghts')
            return __lb_keogh_equal(s1, s2_envelope, use_c)
        else:
            logger.info('Computing LB Keogh distance between two series of unequal lenghts')
            return __lb_keogh_unequal(s1, s2, window, use_c)

    s1 = SeriesContainer.wrap(s1)
    if s2 is not None:
        s2 = SeriesContainer.wrap(s2)
    d = len(s1)

    lang = 'C' if use_c else 'Python'
    if use_c and parallel and not use_mp:
        logger.info("Compute LB Keogh distances in C (parallel=OMP)")
        if equal_series:
            lb = dtw_search_cc.lb_keogh_from_envelope_parallel(s1, s2_envelope)
        else:
            lb = dtw_search_cc.lb_keogh_parallel(s1, s2, window=window)
    elif parallel and use_mp:
        logger.info("Compute LB Keogh distances in {} (parallel=MP)".format(lang))
        with mp.Pool() as p:
            if equal_series:
                lb = p.map(__lb_keogh_equal, [(s1[di], s2_envelope[ei], use_c) for (di, ei) in itertools.product(range(len(s1)), range(len(s2_envelope)))])
            else:
                lb = p.map(__lb_keogh_unequal, [(s1[di], s2[ei], window, use_c) for (di, ei) in itertools.product(range(len(s1)), range(len(s2)))])
            lb = np.array(lb).reshape((len(s1), len(s2)))
    else:
        logger.info("Compute LB Keogh distances in {} (parallel=No)".format(lang))
        lb = np.zeros((d, len(s2_envelope) if s2_envelope else len(s2)))
        for di in range(d):
            for ei in range(len(s2_envelope) if s2_envelope else len(s2)):
                if equal_series:
                    lb[di, ei] = __lb_keogh_equal(s1[di], s2_envelope[ei], use_c)
                else:
                    lb[di, ei] = __lb_keogh_unequal(s1[di], s2[ei], window, use_c)

    return lb


def nearest_neighbour_lb_keogh_fast(data, target, t_envelope, distParams={}):
    """
    lb_keogh nearest neighbour_fast (1NN)
    See nearest_neighbour_lb_keogh
    """
    return nearest_neighbour_lb_keogh(data, target, t_envelope, distParams, True, True)


def nearest_neighbour_lb_keogh(s, t=None, t_envelope=None, distParams={}, use_c=False, use_parallel=False):
    """
    lb_keogh nearest neighbour (1NN)
    Return 1NN result,s ped up by early stopping and lower bound
    :param data: 2D Array of 1D time series
    :param L: Lower envelope part, calculated by lb_keogh_envelopes
    :param U: Upper envelope part, calculated by lb_keogh_envelopes
    :param distParams: Distance function paraneters. For correctness, 'window' should match the envelope window
    :param use_c: Use fast pure c compiled functions
    :param use_parallel: Use fast parallel version (only in C version)
    Returns: 1D array of nearest neighbours indices from the envelopes
    """
    if use_c:
        if dtw_search_cc is None:
            logger.warning("C-library not available, using the Python version")
            use_c = False

    s = SeriesContainer.wrap(s)
    t = SeriesContainer.wrap(t)
    d = len(s)

    if use_c:
        best_fits = array.array('i', [-1]*len(t))
        for ti in range(len(t)):
            loc, dist = dtw_search_cc.nearest_neighbour_lb_keogh(s, t[ti], **distParams)
            print(loc, dist)
            best_fits[ti] = loc
    else:
        lb = lb_keogh(s, t, t_envelope, use_c, use_parallel)
        best_fits = array.array('i', [0]*len(t))

        for ti in range(len(t)):
            best_score_so_far = np.inf
            for di in range(d):
                if best_score_so_far > lb[di, ti]:
                    score = distance(s[di], t[ti], **distParams, max_dist=best_score_so_far)
                    if score < best_score_so_far:
                        best_score_so_far = score
                        best_fits[ti] = di

    return best_fits

