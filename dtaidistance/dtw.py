# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw
~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW)

:author: Wannes Meert
:copyright: Copyright 2017-2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import math
import array

from . import ed
from . import util
from . import util_numpy
from .util import SeriesContainer
from .exceptions import NumpyException


logger = logging.getLogger("be.kuleuven.dtai.distance")


dtw_cc = None
try:
    from . import dtw_cc
except ImportError:
    logger.debug('DTAIDistance C library not available')
    dtw_cc = None

dtw_cc_omp = None
try:
    from . import dtw_cc_omp
except ImportError:
    logger.debug('DTAIDistance C-OMP library not available')
    dtw_cc_omp = None

dtw_cc_numpy = None
try:
    from . import dtw_cc_numpy
except ImportError:
    logger.debug('DTAIDistance C-Numpy library not available')
    dtw_cc_numpy = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
    DTYPE = np.double
    argmin = np.argmin
    array_min = np.min
    array_max = np.max
except ImportError:
    np = None
    argmin = util.argmin
    array_min = min
    array_max = max


def try_import_c():
    global dtw_cc
    try:
        from . import dtw_cc
    except ImportError as exc:
        print('Cannot import C library')
        print(exc)
        dtw_cc = None


inf = float("inf")


def _check_library(include_omp=False, raise_exception=True):
    if dtw_cc is None:
        msg = "The compiled dtaidistance C library is not available.\n" + \
              "See the documentation for alternative installation options."
        logger.error(msg)
        if raise_exception:
            raise Exception(msg)
    if include_omp and dtw_cc_omp is None:
        msg = "The compiled dtaidistance C-OMP library is not available.\n" + \
              "See the documentation for alternative installation options."
        logger.error(msg)
        if raise_exception:
            raise Exception(msg)


def lb_keogh(s1, s2, window=None, max_dist=None,
             max_step=None, max_length_diff=None):
    """Lowerbound LB_KEOGH"""
    # TODO: This implementation slower than distance() in C
    if window is None:
        window = max(len(s1), len(s2))

    t = 0
    for i in range(len(s1)):
        imin = max(0, i - max(0, len(s1) - len(s2)) - window + 1)
        imax = min(len(s2), i + max(0, len(s2) - len(s1)) + window)
        ui = array_max(s2[imin:imax])
        li = array_min(s2[imin:imax])
        ci = s1[i]
        if ci > ui:
            t += abs(ci - ui)
        elif ci < li:
            t += abs(ci - li)
        else:
            pass
    return t


def ub_euclidean(s1, s2):
    """ See ed.euclidean_distance"""
    return ed.distance(s1, s2)


def distance(s1, s2,
             window=None, max_dist=None, max_step=None,
             max_length_diff=None, penalty=None, psi=None,
             use_c=False, use_pruning=False, only_ub=False):
    """
    Dynamic Time Warping.

    This function keeps a compact matrix, not the full warping paths matrix.

    :param s1: First sequence
    :param s2: Second sequence
    :param window: Only allow for maximal shifts from the two diagonals smaller than this number.
        It includes the diagonal, meaning that an Euclidean distance is obtained by setting weight=1.
    :param max_dist: Stop if the returned values will be larger than this value
    :param max_step: Do not allow steps larger than this value
    :param max_length_diff: Return infinity if length of two series is larger
    :param penalty: Penalty to add if compression or expansion is applied
    :param psi: Psi relaxation parameter (ignore start and end of matching).
        Useful for cyclical series.
    :param use_c: Use fast pure c compiled functions
    :param use_pruning: Prune values based on Euclidean distance.
        This is the same as passing ub_euclidean() to max_dist
    :param only_ub: Only compute the upper bound (Euclidean).

    Returns: DTW distance
    """
    if use_c:
        if dtw_cc is None:
            logger.warning("C-library not available, using the Python version")
        else:
            return distance_fast(s1, s2, window,
                                 max_dist=max_dist,
                                 max_step=max_step,
                                 max_length_diff=max_length_diff,
                                 penalty=penalty,
                                 psi=psi,
                                 use_pruning=use_pruning,
                                 only_ub=only_ub)
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
        max_dist = ub_euclidean(s1, s2)**2
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
        for ii in range(i1*length, i1*length+length):
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
            d = (s1[i] - s2[j])**2
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
    Thus ``numpy.array([1,2,3], dtype=numpy.double)`` or
    ``array.array('d', [1,2,3])``
    """
    _check_library(raise_exception=True)
    # Check that Numpy arrays for C contiguous
    s1 = util_numpy.verify_np_array(s1)
    s2 = util_numpy.verify_np_array(s2)
    # Move data to C library
    d = dtw_cc.distance(s1, s2,
                        window=window,
                        max_dist=max_dist,
                        max_step=max_step,
                        max_length_diff=max_length_diff,
                        penalty=penalty,
                        psi=psi,
                        use_pruning=use_pruning,
                        only_ub=only_ub)
    return d


def _distance_with_params(t):
    return distance(t[0], t[1], **t[2])


def _distance_c_with_params(t):
    return dtw_cc.distance(t[0], t[1], **t[2])


def warping_paths(s1, s2, window=None, max_dist=None,
                  max_step=None, max_length_diff=None, penalty=None, psi=None):
    """
    Dynamic Time Warping.

    The full matrix of all warping paths is build.

    :param s1: First sequence
    :param s2: Second sequence
    :param window: see :meth:`distance`
    :param max_dist: see :meth:`distance`
    :param max_step: see :meth:`distance`
    :param max_length_diff: see :meth:`distance`
    :param penalty: see :meth:`distance`
    :param psi: see :meth:`distance`
    :returns: (DTW distance, DTW matrix)
    """
    if np is None:
        raise NumpyException("Numpy is required for the warping_paths method")
    r, c = len(s1), len(s2)
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        return inf
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = inf
    else:
        max_step *= max_step
    if not max_dist:
        max_dist = inf
    else:
        max_dist *= max_dist
    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    if psi is None:
        psi = 0
    dtw = np.full((r + 1, c + 1), inf)
    # dtw[0, 0] = 0
    for i in range(psi + 1):
        dtw[0, i] = 0
        dtw[i, 0] = 0
    i0 = 1
    i1 = 0
    sc = 0
    ec = 0
    smaller_found = False
    ec_next = 0
    for i in range(r):
        i0 = i
        i1 = i + 1
        j_start = max(0, i - max(0, r - c) - window + 1)
        j_end = min(c, i + max(0, c - r) + window)
        if sc > j_start:
            j_start = sc
        smaller_found = False
        ec_next = i
        # print('i =', i, 'skip =',skip, 'skipp =', skipp)
        # jmin = max(0, i - max(0, r - c) - window + 1)
        # jmax = min(c, i + max(0, c - r) + window)
        # print(i,jmin,jmax)
        # x = dtw[i, jmin-skipp:jmax-skipp]
        # y = dtw[i, jmin+1-skipp:jmax+1-skipp]
        # print(x,y,dtw[i+1, jmin+1-skip:jmax+1-skip])
        # dtw[i+1, jmin+1-skip:jmax+1-skip] = np.minimum(x,
        #                                                y)
        for j in range(j_start, j_end):
            # print('j =', j, 'max=',min(c, c - r + i + window))
            d = (s1[i] - s2[j])**2
            if max_step is not None and d > max_step:
                continue
            # print(i, j + 1 - skip, j - skipp, j + 1 - skipp, j - skip)
            dtw[i1, j + 1] = d + min(dtw[i0, j],
                                     dtw[i0, j + 1] + penalty,
                                     dtw[i1, j] + penalty)
            # dtw[i + 1, j + 1 - skip] = d + min(dtw[i + 1, j + 1 - skip], dtw[i + 1, j - skip])
            if dtw[i1, j + 1] > max_dist:
                if not smaller_found:
                    sc = j + 1
                if j >= ec:
                    break
            else:
                smaller_found = True
                ec_next = j + 1
    dtw = np.sqrt(dtw)
    if psi == 0:
        d = dtw[i1, min(c, c + window - 1)]
    else:
        ir = i1
        ic = min(c, c + window - 1)
        vr = dtw[ir-psi:ir+1, ic]
        vc = dtw[ir, ic-psi:ic+1]
        mir = argmin(vr)
        mic = argmin(vc)
        if vr[mir] < vc[mic]:
            dtw[ir-psi+mir+1:ir+1, ic] = -1
            d = vr[mir]
        else:
            dtw[ir, ic - psi + mic + 1:ic+1] = -1
            d = vc[mic]
    if max_dist and d > max_dist:
        d = inf
    return d, dtw


def warping_paths_fast(s1, s2, window=None, max_dist=None,
                       max_step=None, max_length_diff=None, penalty=None, psi=None):
    """Fast C version of :meth:`warping_paths`."""
    s1 = util_numpy.verify_np_array(s1)
    s2 = util_numpy.verify_np_array(s2)
    r = len(s1)
    c = len(s2)
    _check_library(raise_exception=True)
    if window is None:
        window = 0
    if max_dist is None:
        max_dist = 0
    if max_step is None:
        max_step = 0
    if max_length_diff is None:
        max_length_diff = 0
    if penalty is None:
        penalty = 0
    if psi is None:
        psi = 0
    dtw = np.full((r + 1, c + 1), inf)
    d = dtw_cc.warping_paths(dtw, s1, s2,
                             window=window,
                             max_dist=max_dist,
                             max_step=max_step,
                             max_length_diff=max_length_diff,
                             penalty=penalty,
                             psi=psi)
    return d, dtw


def distance_matrix_func(use_c=False, parallel=False, show_progress=False):
    def distance_matrix_wrapper(seqs, **kwargs):
        return distance_matrix(seqs, parallel=parallel, use_c=use_c,
                               show_progress=show_progress, **kwargs)
    return distance_matrix_wrapper


def distance_matrix(s, max_dist=None, use_pruning=False, max_length_diff=None,
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

    logger.info('Computing distances')
    if use_c and parallel and not use_mp and dtw_cc_omp is not None:
        logger.info("Compute distances in C (parallel=OMP)")
        dist_opts['block'] = block
        dists = dtw_cc_omp.distance_matrix(s, **dist_opts)

    elif use_c and parallel and (dtw_cc_omp is None or use_mp):
        logger.info("Compute distances in C (parallel=MP)")
        idxs = _distance_matrix_idxs(block, len(s))
        with mp.Pool() as p:
            dists = p.map(_distance_c_with_params, [(s[r], s[c], dist_opts) for c, r in zip(*idxs)])

    elif use_c and not parallel:
        logger.info("Compute distances in C (parallel=No)")
        dist_opts['block'] = block
        dists = dtw_cc.distance_matrix(s, **dist_opts)

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


def distances_array_to_matrix(dists, nb_series, block=None):
    """Transform a condensed distances array to a full matrix representation.

    The upper triangular matrix will contain all the distances.
    """
    if np is None:
        raise NumpyException("Numpy is required for the distances_array_to_matrix method")
    dists_matrix = np.full((nb_series, nb_series), inf, dtype=DTYPE)
    idxs = _distance_matrix_idxs(block, nb_series)
    dists_matrix[idxs] = dists
    # dists_cond = np.zeros(self._size_cond(len(series)))
    # idx = 0
    # for r in range(len(series) - 1):
    #     dists_cond[idx:idx + len(series) - r - 1] = dists[r, r + 1:]
    #     idx += len(series) - r - 1
    return dists_matrix


def distance_array_index(a, b, nb_series):
    if a == b:
        raise ValueError("Distance between the same series is not available.")
    if a > b:
        a, b = b, a
    idx = 0
    for r in range(a):
        idx += nb_series - r - 1
    idx += b - a - 1
    return idx


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


def _distance_matrix_idxs(block, nb_series):
    if block is None or block == 0:
        if np is not None:
            idxs = np.triu_indices(nb_series, k=1)
            return idxs
        # Numpy is not available
        block = ((0, nb_series), (0, nb_series))
    idxsl_r = []
    idxsl_c = []
    for r in range(block[0][0], block[0][1]):
        for c in range(max(r + 1, block[1][0]), min(nb_series, block[1][1])):
            idxsl_r.append(r)
            idxsl_c.append(c)
    if np is not None:
        idxs = (np.array(idxsl_r), np.array(idxsl_c))
    else:
        idxs = (idxsl_r, idxsl_c)
    return idxs


def _distance_matrix_length(block, nb_series):
    if block is not None:
        block_rb = block[0][0]
        block_re = block[0][1]
        block_cb = block[1][0]
        block_ce = block[1][1]
        length = 0
        for ri in range(block_rb, block_re):
            if block_cb <= ri:
                if block_ce > ri:
                    length += (block_ce - ri - 1)
            else:
                if block_ce > ri:
                    length += (block_ce - block_cb)
    else:
        length = int(nb_series * (nb_series - 1) / 2)
    return length


def distance_matrix_fast(s, max_dist=None, max_length_diff=None,
                         window=None, max_step=None, penalty=None, psi=None,
                         block=None, compact=False, parallel=True):
    """Fast C version of :meth:`distance_matrix`."""
    _check_library(raise_exception=True, include_omp=parallel)
    return distance_matrix(s, max_dist=max_dist, max_length_diff=max_length_diff,
                           window=window, max_step=max_step, penalty=penalty, psi=psi,
                           block=block, compact=compact, parallel=parallel,
                           use_c=True, show_progress=False)


def warping_path(from_s, to_s, **kwargs):
    """Compute warping path between two sequences."""
    dist, paths = warping_paths(from_s, to_s, **kwargs)
    path = best_path(paths)
    return path


def warping_amount(path):
    """
        Returns the number of compressions and expansions performed to obtain the best path.
        Can be used as a metric for the amount of warping.

        :param path: path to be tested

        :returns number of compressions or expansions

    """
    n = 0
    for i in range(1, len(path)):
        if path[i - 1][0] + 1 != path[i][0] or path[i - 1][1] + 1 != path[i][1]:
            n += 1

    return n


def warping_path_penalty(s1, s2, penalty_post=0, **kwargs):
    """Dynamic Time Warping with an alternative penalty.

    This function supports two different penalties. The traditional DTW penalty `penalty`
    is used in the matrix during calculation of the warping path (see :meth:`distance`).

    The second penalty `penalty_post` measures the amount of warping. This penalty doesn't
    affect the warping path and is added to the DTW distance after the warping for every compression or expansion.

    Same options as :meth:`warping_paths`

    :param s1: First sequence
    :param s2: Second sequence
    :param penalty_post: Penalty to be added after path calculation, for compression/extension

    :returns [DTW distance, best path, DTW distance between 2 path elements, DTW matrix]
    """
    dist, paths = warping_paths(s1, s2, **kwargs)
    path = best_path(paths)

    path_stepsize = []
    for i in range(1, len(path)):
        if path[i - 1][0] + 1 != path[i][0] or path[i - 1][1] + 1 != path[i][1]:
            dist += penalty_post

        path_stepsize.append(paths[path[i][0] + 1, path[i][1] + 1] - paths[path[i - 1][0] + 1, path[i - 1][1] + 1])

    return [dist, path, path_stepsize, paths]


def warp(from_s, to_s, path=None, **kwargs):
    """Warp a function to optimally match a second function.

    :param from_s: First sequence
    :param to_s: Second sequence
    :param path: (Optional) Path to use wrap the 'from_s' sequence to the 'to_s' sequence
                If provided, this function will use it.
                If not provided, this function will calculate it using the warping_path function
    :param kwargs: Same options as :meth:`warping_paths`.
    """
    if path is None:
        path = warping_path(from_s, to_s, **kwargs)
    from_s2 = array.array('d', [0] * len(to_s))
    from_s2_cnt = array.array('i', [0] * len(to_s))
    for r_c, c_c in path:
        from_s2[c_c] += from_s[r_c]
        from_s2_cnt[c_c] += 1
    for i in range(len(to_s)):
        from_s2[i] /= from_s2_cnt[i]
    return from_s2, path


def best_path(paths):
    """Compute the optimal path from the nxm warping paths matrix."""
    i, j = int(paths.shape[0] - 1), int(paths.shape[1] - 1)
    p = []
    if paths[i, j] != -1:
        p.append((i - 1, j - 1))
    while i > 0 and j > 0:
        c = argmin([paths[i - 1, j - 1], paths[i - 1, j], paths[i, j - 1]])
        if c == 0:
            i, j = i - 1, j - 1
        elif c == 1:
            i = i - 1
        elif c == 2:
            j = j - 1
        if paths[i, j] != -1:
            p.append((i - 1, j - 1))
    p.pop()
    p.reverse()
    return p


def best_path2(paths):
    """Compute the optimal path from the nxm warping paths matrix."""
    m = paths
    path = []
    r, c = m.shape
    r -= 1
    c -= 1
    v = m[r, c]
    path.append((r - 1, c - 1))
    while r > 1 or c > 1:
        r_c, c_c = r, c
        if r >= 1 and c >= 1 and m[r - 1, c - 1] <= v:
            r_c, c_c, v = r - 1, c - 1, m[r - 1, c - 1]
        if r >= 1 and m[r - 1, c] <= v:
            r_c, c_c, v = r - 1, c, m[r - 1, c]
        if c >= 1 and m[r, c - 1] <= v:
            r_c, c_c, v = r, c - 1, m[r, c - 1]
        path.append((r_c - 1, c_c - 1))
        r, c = r_c, c_c
    path.reverse()
    return path
