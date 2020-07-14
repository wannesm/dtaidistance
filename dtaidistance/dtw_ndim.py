# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw_ndim
~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW) for N-dimensional series.

:author: Wannes Meert
:copyright: Copyright 2017-2018 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import os
import logging
import math
import array

from .dtw import _check_library, SeriesContainer, _distance_matrix_idxs, distances_array_to_matrix,\
    _distance_matrix_length
from . import util_numpy
from .exceptions import NumpyException

try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
    array_min = np.min
except ImportError:
    np = None
    array_min = min


logger = logging.getLogger("be.kuleuven.dtai.distance")
dtaidistance_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)

try:
    from . import dtw_cc
except ImportError:
    # logger.info('C library not available')
    dtw_cc = None

dtw_cc_omp = None
try:
    from . import dtw_cc_omp
except ImportError:
    logger.debug('DTAIDistance C-OMP library not available')
    dtw_cc_omp = None

try:
    from tqdm import tqdm
except ImportError:
    logger.info('tqdm library not available')
    tqdm = None


inf = float("inf")


def ub_euclidean(s1, s2):
    """ Euclidean distance between two n-dimensional sequences. Supports different lengths.

    If the two series differ in length, compare the last element of the shortest series
    to the remaining elements in the longer series.

    :param s1: Sequence of numbers, 1st dimension is sequence, 2nd dimension is n-dimensional value vector.
    :param s2: Sequence of numbers, 1st dimension is sequence, 2nd dimension is n-dimensional value vector.
    :return: Euclidean distance
    """
    if np is None:
        raise NumpyException("Numpy is required for the ub_euclidean method.")
    n = min(len(s1), len(s2))
    ub = 0
    for i in range(n):
        ub += np.sum((s1[i] - s2[i]) ** 2)
    # If the two series differ in length, compare the last element of the shortest series
    # to the remaining elements in the longer series
    if len(s1) > len(s2):
        for i in range(n, len(s1)):
            ub += np.sum((s1[i] - s2[n - 1]) ** 2)
    elif len(s1) < len(s2):
        for i in range(n, len(s2)):
            ub += np.sum((s1[n - 1] - s2[i]) ** 2)
    return math.sqrt(ub)


def distance(s1, s2, window=None, max_dist=None,
             max_step=None, max_length_diff=None, penalty=None, psi=None,
             use_c=False, use_pruning=False, only_ub=False):
    """Dynamic Time Warping using multidimensional sequences.

    Assumes first dimension to be the sequence item index, and the second
    dimension to be the dimension in the sequence item.

    cost = EuclideanDistance(s1[i], s2[j])

    See :py:meth:`dtaidistance.dtw.distance` for parameters.
    """
    if use_c:
        if dtw_cc is None:
            logger.warning("C-library not available, using the Python version")
        else:
            return distance_fast(s1, s2,
                                 window=window,
                                 max_dist=max_dist,
                                 max_step=max_step,
                                 max_length_diff=max_length_diff,
                                 penalty=penalty,
                                 psi=psi,
                                 use_pruning=use_pruning,
                                 only_ub=only_ub)
    if np is None:
        raise NumpyException("Numpy is required for the dtw_ndim.distance method "
                             "(Numpy is not required for the distance_fast method that uses the C library")
    r, c = len(s1), len(s2)
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        return inf
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = inf
    else:
        max_step *= max_step
    if use_pruning or only_ub:
        max_dist = ub_euclidean(s1, s2) ** 2
        if only_ub:
            return max_dist
    elif not max_dist:
        max_dist = inf
    else:
        max_dist *= max_dist
    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    if psi is None:
        psi = 0
    length = min(c + 1, abs(r - c) + 2 * (window - 1) + 1 + 1 + 1)
    # print("length (py) = {}".format(length))
    dtw = array.array('d', [inf] * (2 * length))
    sc = 0
    ec = 0
    ec_next = 0
    smaller_found = False
    for i in range(psi + 1):
        dtw[i] = 0
    skip = 0
    i0 = 1
    i1 = 0
    psi_shortest = inf
    for i in range(r):
        # print("i={}".format(i))
        # print(dtw)
        skipp = skip
        skip = max(0, i - max(0, r - c) - window + 1)
        i0 = 1 - i0
        i1 = 1 - i1
        for ii in range(i1 * length, i1 * length + length):
            dtw[ii] = inf
        j_start = max(0, i - max(0, r - c) - window + 1)
        j_end = min(c, i + max(0, c - r) + window)
        if sc > j_start:
            j_start = sc
        smaller_found = False
        ec_next = i
        if length == c + 1:
            skip = 0
        if psi != 0 and j_start == 0 and i < psi:
            dtw[i1 * length] = 0
        for j in range(j_start, j_end):
            d = np.sum((s1[i] - s2[j]) ** 2)
            if d > max_step:
                continue
            assert j + 1 - skip >= 0
            assert j - skipp >= 0
            assert j + 1 - skipp >= 0
            assert j - skip >= 0
            dtw[i1 * length + j + 1 - skip] = d + min(dtw[i0 * length + j - skipp],
                                                      dtw[i0 * length + j + 1 - skipp] + penalty,
                                                      dtw[i1 * length + j - skip] + penalty)
            # print('({},{}), ({},{}), ({},{})'.format(i0, j - skipp, i0, j + 1 - skipp, i1, j - skip))
            # print('{}, {}, {}'.format(dtw[i0, j - skipp], dtw[i0, j + 1 - skipp], dtw[i1, j - skip]))
            # print('i={}, j={}, d={}, skip={}, skipp={}'.format(i,j,d,skip,skipp))
            # print(dtw)
            if dtw[i1 * length + j + 1 - skip] > max_dist:
                if not smaller_found:
                    sc = j + 1
                if j >= ec:
                    break
            else:
                smaller_found = True
                ec_next = j + 1
        ec = ec_next
        if psi != 0 and j_end == len(s2) and len(s1) - 1 - i <= psi:
            psi_shortest = min(psi_shortest, dtw[i1 * length + length - 1])
    if psi == 0:
        d = math.sqrt(dtw[i1 * length + min(c, c + window - 1) - skip])
    else:
        ic = min(c, c + window - 1) - skip
        vc = dtw[i1 * length + ic - psi:i1 * length + ic + 1]
        d = min(array_min(vc), psi_shortest)
        d = math.sqrt(d)
    if max_dist and d > max_dist:
        d = inf
    return d


def distance_fast(s1, s2, window=None, max_dist=None,
                  max_step=None, max_length_diff=None, penalty=None, psi=None, use_pruning=False, only_ub=False):
    """Fast C version of :meth:`distance`.

    Note: the series are expected to be arrays of the type ``double``.
    Thus ``numpy.array([[1,1],[2,2],[3,3]], dtype=numpy.double)``    """
    _check_library(raise_exception=True)
    # Check that Numpy arrays for C contiguous
    if np is not None:
        if isinstance(s1, (np.ndarray, np.generic)):
            if not s1.data.c_contiguous:
                logger.debug("Warning: Sequence 1 passed to method distance is not C-contiguous. " +
                             "The sequence will be copied.")
                s1 = s1.copy(order='C')
        if isinstance(s2, (np.ndarray, np.generic)):
            if not s2.data.c_contiguous:
                logger.debug("Warning: Sequence 2 passed to method distance is not C-contiguous. " +
                             "The sequence will be copied.")
                s2 = s2.copy(order='C')
    # Move data to C library
    d = dtw_cc.distance_ndim(s1, s2,
                             window=window,
                             max_dist=max_dist,
                             max_step=max_step,
                             max_length_diff=max_length_diff,
                             penalty=penalty,
                             psi=psi,
                             use_pruning=use_pruning,
                             only_ub=only_ub)
    return d


def warping_paths(s1, s2, window=None, max_dist=None,
                  max_step=None, max_length_diff=None, penalty=None, psi=None,):
    """
    Dynamic Time Warping (keep full matrix) using multidimensional sequences.

    cost = EuclideanDistance(s1[i], s2[j])

    See :py:meth:`dtaidistance.dtw.warping_paths` for parameters.
    """
    if np is None:
        raise NumpyException("Numpy is required for the warping_paths method")
    r, c = len(s1), len(s2)
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        return np.inf
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = np.inf
    else:
        max_step *= max_step
    if not max_dist:
        max_dist = np.inf
    else:
        max_dist *= max_dist
    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    if psi is None:
        psi = 0
    dtw = np.full((r + 1, c + 1), np.inf)
    # dtw[0, 0] = 0
    for i in range(psi + 1):
        dtw[0, i] = 0
        dtw[i, 0] = 0
    last_under_max_dist = 0
    i0 = 1
    i1 = 0
    for i in range(r):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = np.inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        i0 = i
        i1 = i + 1
        # print('i =', i, 'skip =',skip, 'skipp =', skipp)
        # jmin = max(0, i - max(0, r - c) - window + 1)
        # jmax = min(c, i + max(0, c - r) + window)
        # print(i,jmin,jmax)
        # x = dtw[i, jmin-skipp:jmax-skipp]
        # y = dtw[i, jmin+1-skipp:jmax+1-skipp]
        # print(x,y,dtw[i+1, jmin+1-skip:jmax+1-skip])
        # dtw[i+1, jmin+1-skip:jmax+1-skip] = np.minimum(x,
        #                                                y)
        for j in range(max(0, i - max(0, r - c) - window + 1), min(c, i + max(0, c - r) + window)):
            # print('j =', j, 'max=',min(c, c - r + i + window))
            d = np.sum((s1[i] - s2[j]) ** 2)
            if max_step is not None and d > max_step:
                continue
            # print(i, j + 1 - skip, j - skipp, j + 1 - skipp, j - skip)
            dtw[i1, j + 1] = d + min(dtw[i0, j],
                                     dtw[i0, j + 1] + penalty,
                                     dtw[i1, j] + penalty)
            # dtw[i + 1, j + 1 - skip] = d + min(dtw[i + 1, j + 1 - skip], dtw[i + 1, j - skip])
            if max_dist is not None:
                if dtw[i1, j + 1] <= max_dist:
                    last_under_max_dist = j
                else:
                    dtw[i1, j + 1] = np.inf
                    if prev_last_under_max_dist < j + 1:
                        break
        if max_dist is not None and last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            return np.inf, dtw
    dtw = np.sqrt(dtw)
    if psi == 0:
        d = dtw[i1, min(c, c + window - 1)]
    else:
        ir = i1
        ic = min(c, c + window - 1)
        vr = dtw[ir-psi:ir+1, ic]
        vc = dtw[ir, ic-psi:ic+1]
        mir = np.argmin(vr)
        mic = np.argmin(vc)
        if vr[mir] < vc[mic]:
            dtw[ir-psi+mir+1:ir+1, ic] = -1
            d = vr[mir]
        else:
            dtw[ir, ic - psi + mic + 1:ic+1] = -1
            d = vc[mic]
    return d, dtw


def _distance_with_params(t):
    return distance(t[0], t[1], **t[2])


def _distance_c_with_params(t):
    return dtw_cc.distance_ndim(t[0], t[1], **t[3])


def distance_matrix_python(s, block=None, show_progress=False, max_length_diff=None, dist_opts=None):
    if dist_opts is None:
        dist_opts = {}
    dists = array.array('d', [inf] * _distance_matrix_length(block, len(s)))
    if block is None:
        it_r = range(len(s))
    else:
        it_r = range(block[0][0], block[0][1])
    if show_progress:
        it_r = tqdm(it_r)
    idx = 0
    for r in it_r:
        if block is None:
            it_c = range(r + 1, len(s))
        else:
            it_c = range(max(r + 1, block[1][0]), min(len(s), block[1][1]))
        for c in it_c:
            if abs(len(s[r]) - len(s[c])) <= max_length_diff:
                dists[idx] = distance(s[r], s[c], **dist_opts)
            idx += 1
    return dists


def distance_matrix(s, ndim, max_dist=None, use_pruning=False, max_length_diff=None,
                    window=None, max_step=None, penalty=None, psi=None,
                    block=None, compact=False, parallel=False,
                    use_c=False, use_mp=False, show_progress=False):
    """Distance matrix for all sequences in s.

    :param s: Iterable of series
    :param window: see :meth:`distance`
    :param max_dist: see :meth:`distance`
    :param max_step: see :meth:`distance`
    :param max_length_diff: see :meth:`distance`
    :param penalty: see :meth:`distance`
    :param psi: see :meth:`distance`
    :param block: Only compute block in matrix. Expects tuple with begin and end, e.g. ((0,10),(20,25)) will
        only compare rows 0:10 with rows 20:25.
    :param compact: Return the distance matrix as an array representing the upper triangular matrix.
    :param parallel: Use parallel operations
    :param use_c: Use c compiled Python functions
    :param use_mp: Use Multiprocessing for parallel operations (not OpenMP)
    :param show_progress: Show progress using the tqdm library. This is only supported for
        the pure Python version (thus not the C-based implementations).
    :returns: The distance matrix or the condensed distance matrix if the compact argument is true
    """
    # Check whether multiprocessing is available
    if use_c:
        _check_library(raise_exception=True)
    if use_c and parallel:
        if dtw_cc_omp is None:
            logger.warning('OMP extension not loaded, using multiprocessing')
    if parallel and (use_mp or not use_c or dtw_cc_omp is None):
        try:
            import multiprocessing as mp
            logger.info('Using multiprocessing')
        except ImportError:
            msg = 'Cannot load multiprocessing'
            logger.error(msg)
            raise Exception(msg)
    else:
        mp = None
    # Prepare options and data to pass to distance method
    dist_opts = {
        'max_dist': max_dist,
        'max_step': max_step,
        'window': window,
        'max_length_diff': max_length_diff,
        'penalty': penalty,
        'psi': psi,
        'use_pruning': use_pruning
    }
    s = SeriesContainer.wrap(s)
    if max_length_diff is None:
        max_length_diff = inf
    dists = None
    if use_c:
        for k, v in dist_opts.items():
            if v is None:
                # None is represented as 0.0 for C
                dist_opts[k] = 0.0

    logger.info('Computing n-dim distances')
    if use_c and parallel and not use_mp and dtw_cc_omp is not None:
        logger.info("Compute distances in C (parallel=OMP)")
        dist_opts['block'] = block
        dists = dtw_cc_omp.distance_matrix_ndim(s, ndim, **dist_opts)

    elif use_c and parallel and (dtw_cc_omp is None or use_mp):
        logger.info("Compute distances in C (parallel=MP)")
        idxs = _distance_matrix_idxs(block, len(s))
        with mp.Pool() as p:
            dists = p.map(_distance_c_with_params, [(s[r], s[c], dist_opts) for c, r in zip(*idxs)])

    elif use_c and not parallel:
        logger.info("Compute distances in C (parallel=No)")
        dist_opts['block'] = block
        dists = dtw_cc.distance_matrix_ndim(s, ndim, **dist_opts)

    elif not use_c and parallel:
        logger.info("Compute distances in Python (parallel=MP)")
        idxs = _distance_matrix_idxs(block, len(s))
        with mp.Pool() as p:
            dists = p.map(_distance_with_params, [(s[r], s[c], dist_opts) for c, r in zip(*idxs)])

    elif not use_c and not parallel:
        logger.info("Compute distances in Python (parallel=No)")
        dists = distance_matrix_python(s, block=block, show_progress=show_progress,
                                       max_length_diff=max_length_diff, dist_opts=dist_opts)

    else:
        raise Exception(f'Unsupported combination of: parallel={parallel}, '
                        f'use_c={use_c}, dtw_cc_omp={dtw_cc_omp}, use_mp={use_mp}')

    exp_length = _distance_matrix_length(block, len(s))
    assert len(dists) == exp_length, "len(dists)={} != {}".format(len(dists), exp_length)
    if compact:
        return dists

    # Create full matrix and fill upper triangular matrix with distance values (or only block if specified)
    dists_matrix = distances_array_to_matrix(dists, nb_series=len(s), block=block)

    return dists_matrix


def distance_matrix_fast(s, ndim, max_dist=None, max_length_diff=None,
                         window=None, max_step=None, penalty=None, psi=None,
                         block=None, compact=False, parallel=True):
    """Fast C version of :meth:`distance_matrix`."""
    _check_library(raise_exception=True, include_omp=parallel)
    return distance_matrix(s, ndim, max_dist=max_dist, max_length_diff=max_length_diff,
                           window=window, max_step=max_step, penalty=penalty, psi=psi,
                           block=block, compact=compact, parallel=parallel,
                           use_c=True, show_progress=False)