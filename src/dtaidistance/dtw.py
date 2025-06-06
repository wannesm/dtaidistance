# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw
~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW)

:author: Wannes Meert
:copyright: Copyright 2017-2024 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import array
import math
from argparse import ArgumentError

from . import ed
from . import util
from . import util_numpy
from . import innerdistance
from .util import SeriesContainer
from .exceptions import NumpyException, CythonException


logger = logging.getLogger("be.kuleuven.dtai.distance")

dtw_cc = None
try:
    from . import dtw_cc
except ImportError:
    logger.debug('DTAIDistance C library not available')
    dtw_cc = None

dtw_cc_omp = None
dtw_cc_omp_err = None
try:
    from . import dtw_cc_omp
except ImportError as exc:
    dtw_cc_omp_err = str(exc)
    logger.debug('DTAIDistance C-OMP library not available')
    logger.debug(exc)
    dtw_cc_omp = None

dtw_cc_numpy = None
try:
    from . import dtw_cc_numpy
except ImportError:
    logger.debug('DTAIDistance C-Numpy library not available')
    dtw_cc_numpy = None
except ValueError as exc:
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
    argmax = np.argmax
    array_min = np.min
    array_max = np.max
except ImportError:
    np = None
    argmin = util.argmin
    argmax = util.argmax
    array_min = min
    array_max = max


def try_import_c(verbose=False):
    return util.try_import_c(verbose)


inf = float("inf")


def _check_library(include_omp=False, raise_exception=True):
    if dtw_cc is None:
        msg = "The compiled dtaidistance C library is not available.\n" + \
              "See the documentation for alternative installation options."
        logger.error(msg)
        if raise_exception:
            raise CythonException(msg)
    if include_omp and (dtw_cc_omp is None or not dtw_cc_omp.is_openmp_supported()):
        msg = "The compiled dtaidistance C-OMP library "
        if dtw_cc_omp and not dtw_cc_omp.is_openmp_supported():
            msg += "indicates that OpenMP was not avaiable during compilation.\n"
        else:
            msg += "is not available.\n"
        msg += "Use Python's multiprocessing library for parellelization (use_mp=True).\n" + \
               "Call dtw.try_import_c() to get more verbose errors.\n" + \
               "See the documentation for alternative installation options."
        logger.error(msg)
        if raise_exception:
            raise CythonException(msg)


class DTWSettings:
    def __init__(self, window=None, use_pruning=False, max_dist=None, max_step=None,
                 max_length_diff=None, penalty=None, psi=None, inner_dist=innerdistance.default,
                 use_ndim=False, use_c=False):
        """Settings for Dynamic Time Warping distance methods.

        :param window: Only allow for maximal shifts from the two diagonals smaller than this number.
            The maximally allowed warping, thus difference between indices i in series 1 and j in series 2,
            is thus ``|i-j| < 2*window + |len(s1) - len(s2)|``.
            It includes the diagonal, meaning that Euclidean distance is obtained by setting
            ``window=1.``
            If the two series are of equal length, this means that the band you see appearing
            on the cumulative cost matrix is of width 2*window-1. In other definitions of DTW
            this is referred to as the window.
        :param max_dist: Stop if the returned values will be larger than this value
        :param max_step: Do not allow steps larger than this value.
            If the difference between two values in the two series is larger than this, thus
            if ``|s1[i]-s2[j]| > max_step``, replace that value with infinity.
        :param max_length_diff: Return infinity if length of two series is larger
        :param penalty: Penalty to add if compression or expansion is applied
        :param psi: Psi relaxation parameter (ignore start and end of matching).
            If psi is a single integer, it is used for both start and end relaxations of both series.
            If psi is a 4-tuple, it is used as the psi-relaxation for
            (begin series1, end series1, begin series2, end series2).
            Useful for cyclical series.
        :param use_pruning: Prune values based on Euclidean distance.
            This is the same as passing ub_euclidean() to max_dist
        :param inner_dist: Distance between two points in the time series.
            One of 'squared euclidean' (default), 'euclidean'.
            When using the pure Python implementation (thus use_c=False) then the argument can also
            be an object that has as callable arguments 'inner_dist', 'result', and 'inner_val'. The 'inner_dist'
            function computes the distance between two points (e.g., squared euclidean) and 'result'
            is the function to apply to the final distance (e.g., sqrt when using squared euclidean).
            You can also inherit from the 'innerdistance.CustomInnerDist' class.
        :param use_ndim: Use n-dimensional (aka multivariate) series instead of 1-dimensional series.
        :param use_c: Use the C variant if available.
        """
        self.window = window
        self.use_pruning = use_pruning
        self.max_dist = max_dist
        self.max_step = max_step
        self.max_length_diff = max_length_diff
        self.penalty = penalty
        self.psi = psi
        self.inner_dist = inner_dist
        self.use_ndim = use_ndim
        self.use_c = use_c

        _, _, inner_val = innerdistance.inner_dist_fns(self.inner_dist)

        if not self.max_step:
            self.adj_max_step = inf
        else:
            self.adj_max_step = inner_val(self.max_step)

        if not self.max_dist:
            self.adj_max_dist = inf
        else:
            self.adj_max_dist = inner_val(self.max_dist)

        if not self.penalty:
            self.adj_penalty = 0
        else:
            self.adj_penalty = inner_val(self.penalty)

        if self.max_length_diff is None:
            self.adj_max_length_diff = inf
        else:
            self.adj_max_length_diff = self.max_length_diff

    @staticmethod
    def wrap(settings):
        if isinstance(settings, DTWSettings):
            return settings
        if settings is None:
            return DTWSettings()
        return DTWSettings(**settings)

    @staticmethod
    def for_dtw(s1, s2, **kwargs):
        settings = DTWSettings(**kwargs)
        if settings.window is None:
            settings.window = max(len(s1), len(s2))
        settings.set_max_dist(s1, s2)
        return settings

    def set_max_dist(self, s1, s2):
        _, _, ival_fn = innerdistance.inner_dist_fns(self.inner_dist, use_ndim=self.use_ndim)
        if self.use_pruning:
            self.adj_max_dist = ival_fn(ub_euclidean(s1, s2, inner_dist=self.inner_dist))

    def kwargs(self):
        return {
            'window': self.window,
            'use_pruning': self.use_pruning,
            'max_dist': self.max_dist,
            'max_step': self.max_step,
            'max_length_diff': self.max_length_diff,
            'penalty': self.penalty,
            'psi': self.psi,
            'inner_dist': self.inner_dist,
            'use_ndim': self.use_ndim,
            'use_c': self.use_c
        }

    def c_kwargs(self):
        window = 0 if self.window is None else self.window
        max_dist = 0 if self.max_dist is None else self.max_dist
        max_step = 0 if self.max_step is None else self.max_step
        max_length_diff = 0 if (self.max_length_diff is None or math.isinf(self.max_length_diff)) \
            else self.max_length_diff
        penalty = 0 if self.penalty is None else self.penalty
        psi = 0 if self.psi is None else self.psi
        use_pruning = 0 if self.use_pruning is None else self.use_pruning
        inner_dist = innerdistance.to_c(self.inner_dist)
        return {
            'window': window,
            'max_dist': max_dist,
            'max_step': max_step,
            'max_length_diff': max_length_diff,
            'penalty': penalty,
            'psi': psi,
            'use_pruning': use_pruning,
            'inner_dist': inner_dist
        }

    def split_psi(self):
        psi_1b = psi_1e = psi_2b = psi_2e = 0
        if type(self.psi) is int:
            psi_1b = psi_1e = psi_2b = psi_2e = self.psi
        elif type(self.psi) in [tuple, list]:
            psi_1b, psi_1e, psi_2b, psi_2e = self.psi
        return psi_1b, psi_1e, psi_2b, psi_2e

    def to_h5_group(self, group):
        for key, value in self.c_kwargs().items():
            group.attrs[key] = value

    @staticmethod
    def from_h5_group(group):
        kwargs = {}
        for attr in ["window", "use_pruning", "max_dist", "max_step", "max_step_clip",
                     "max_length_diff", "penalty", "psi", "inner_dist", "use_ndim", "use_c"]:
            if attr in group.attrs:
                kwargs[attr] = group.attrs[attr]
        return DTWSettings(**kwargs)

    def __str__(self):
        r = ''
        a = self.c_kwargs()
        for k, v in a.items():
            r += '{}: {}\n'.format(k, v)
        return r


def lb_keogh(s1, s2, **kwargs):
    """Lowerbound LB_KEOGH"""
    s = DTWSettings(**kwargs)
    if s.use_c:
        return dtw_cc.lb_keogh(s1, s2, window=s.window, max_dist=s.max_dist,
                               max_step=s.max_step, inner_dist=s.inner_dist)
    if s.window is None:
        s.window = max(len(s1), len(s2))
    idist_fn, result_fn, _ = innerdistance.inner_dist_fns(s.inner_dist, use_ndim=s.use_ndim)

    t = 0
    imin_diff = max(0, len(s1) - len(s2)) + s.window - 1
    imax_diff = max(0, len(s2) - len(s1)) + s.window
    for i in range(len(s1)):
        imin = max(0, i - imin_diff)
        imax = min(len(s2), i + imax_diff)
        ui = array_max(s2[imin:imax])
        li = array_min(s2[imin:imax])
        ci = s1[i]
        if ci > ui:
            t += idist_fn(ci, ui)  # (ci - ui)**2
        elif ci < li:
            t += idist_fn(ci, li)  # (ci - li)**2
        else:
            pass
    return result_fn(t)


def ub_euclidean(s1, s2, inner_dist=innerdistance.default):
    """ See :meth:`dtaidistance.ed.euclidean_distance`"""
    return ed.distance(s1, s2, inner_dist=inner_dist)


def distance(s1, s2, only_ub=False, **kwargs):
    """
    Dynamic Time Warping.

    This function keeps a compact matrix, not the full warping paths matrix.

    Uses dynamic programming to compute::

        wps[i, j] = (s1[i]-s2[j])**2 + min(
                        wps[i-1, j  ] + penalty,  // vertical   / insertion / expansion
                        wps[i  , j-1] + penalty,  // horizontal / deletion  / compression
                        wps[i-1, j-1])            // diagonal   / match
        dtw = sqrt(wps[-1, -1])

    :param s1: First sequence
    :param s2: Second sequence
    :param only_ub: Only compute the upper bound (Euclidean).
    :param kwargs: :class:`DTWSettings` arguments

    Returns: DTW distance
    """
    s = DTWSettings.for_dtw(s1, s2, **kwargs)
    if s.use_c:
        if dtw_cc is None:
            logger.warning("C-library not available, using the Python version")
        else:
            return distance_fast(s1, s2, **s.kwargs())
    idist_fn, result_fn, ival_fn = innerdistance.inner_dist_fns(s.inner_dist, use_ndim=s.use_ndim)
    r, c = len(s1), len(s2)
    if s.adj_max_length_diff is not None and abs(r - c) > s.adj_max_length_diff:
        return inf
    if only_ub:
        return ival_fn(ub_euclidean(s1, s2, inner_dist=s.inner_dist))

    psi_1b, psi_1e, psi_2b, psi_2e = s.split_psi()
    length = min(c + 1, abs(r - c) + 2 * (s.window - 1) + 1 + 1 + 1)
    dtw = array.array('d', [inf] * (2 * length))
    sc = 0
    ec = 0
    for i in range(psi_2b + 1):
        dtw[i] = 0
    skip = 0
    i0 = 1
    i1 = 0
    psi_shortest = inf
    for i in range(r):
        # print("i={}".format(i))
        # print(dtw)
        skipp = skip
        skip = max(0, i - max(0, r - c) - s.window + 1)
        i0 = 1 - i0
        i1 = 1 - i1
        for ii in range(i1*length, i1*length+length):
            dtw[ii] = inf
        j_start = max(0, i - max(0, r - c) - s.window + 1)
        j_end = min(c, i + max(0, c - r) + s.window)
        if sc > j_start:
            j_start = sc
        smaller_found = False
        ec_next = i
        if length == c + 1:
            skip = 0
        if psi_1b != 0 and j_start == 0 and i < psi_1b:
            dtw[i1 * length] = 0
        for j in range(j_start, j_end):
            # d = (s1[i] - s2[j])**2
            d = idist_fn(s1[i], s2[j])
            if d > s.adj_max_step:
                continue
            assert j + 1 - skip >= 0
            assert j - skipp >= 0
            assert j + 1 - skipp >= 0
            assert j - skip >= 0
            dtw[i1 * length + j + 1 - skip] = d + min(dtw[i0 * length + j - skipp],
                                                      dtw[i0 * length + j + 1 - skipp] + s.adj_penalty,
                                                      dtw[i1 * length + j - skip] + s.adj_penalty)
            # print('({},{}), ({},{}), ({},{})'.format(i0, j - skipp, i0, j + 1 - skipp, i1, j - skip))
            # print('{}, {}, {}'.format(dtw[i0, j - skipp], dtw[i0, j + 1 - skipp], dtw[i1, j - skip]))
            # print('i={}, j={}, d={}, skip={}, skipp={}'.format(i,j,d,skip,skipp))
            # print(dtw)
            if dtw[i1 * length + j + 1 - skip] > s.adj_max_dist:
                if not smaller_found:
                    sc = j + 1
                if j >= ec:
                    break
            else:
                smaller_found = True
                ec_next = j + 1
        ec = ec_next
        if psi_1e != 0 and j_end == len(s2) and len(s1) - 1 - i <= psi_1e:
            psi_shortest = min(psi_shortest, dtw[i1 * length + j_end - skip])
    if psi_1e == 0 and psi_2e == 0:
        d = dtw[i1 * length + min(c, c + s.window - 1) - skip]
    else:
        ic = min(c, c + s.window - 1) - skip
        if psi_2e != 0:
            vc = dtw[i1 * length + ic - psi_2e:i1 * length + ic + 1]
            d = min(array_min(vc), psi_shortest)
        else:
            d = min(dtw[i1 * length + min(c, c + s.window - 1) - skip], psi_shortest)
    if s.adj_max_dist and d > s.adj_max_dist:
        d = inf
    d = result_fn(d)
    return d


def distance_fast(s1, s2, only_ub=False, use_pruning=True, **kwargs):
    """Same as :meth:`distance` but with different defaults to choose the fast C-based version of
    the implementation (use_c = True) and use pruning (use_pruning = True).

    Note: the series are expected to be arrays of the type ``double``.
    Thus ``numpy.array([1,2,3], dtype=numpy.double)`` or
    ``array.array('d', [1,2,3])``
    """
    _check_library(raise_exception=True)
    # Check that Numpy arrays for C contiguous
    s1 = util_numpy.verify_np_array(s1)
    s2 = util_numpy.verify_np_array(s2)
    s = DTWSettings(use_pruning=use_pruning, **kwargs)
    # Move data to C library
    if s.use_ndim is False:
        d = dtw_cc.distance(s1, s2, only_ub=only_ub, **s.c_kwargs())
    else:
        d = dtw_cc.distance_ndim(s1, s2, only_ub=only_ub, **s.c_kwargs())
    return d


def _distance_with_params(t):
    return distance(t[0], t[1], **t[2])


def _distance_with_params_ndim(t):
    return distance(t[0], t[1], use_ndim=True, **t[2])


def _distance_c_with_params(t):
    return dtw_cc.distance(t[0], t[1], **t[2])


def _distance_c_with_params_ndim(t):
    return dtw_cc.distance_ndim(t[0], t[1], **t[2])


def warping_paths(s1, s2, psi_neg=True, keep_int_repr=False, **kwargs):
    """
    Dynamic Time Warping.

    The full matrix of all warping paths (or accumulated cost matrix) is built.

    :param s1: First sequence
    :param s2: Second sequence
    :param psi_neg: Replace values that should be skipped because of psi-relaxation with -1.
    :param keep_int_repr: Keep the internal representation of the numbers.
        For example, by default this is the squared differences.
    :param kwargs: See arguments for :class:`DTWSettings`
    :returns: (DTW distance, DTW matrix)
    """
    s = DTWSettings.for_dtw(s1, s2, **kwargs)
    if s.use_c:
        return warping_paths_fast(s1, s2, psi_neg=psi_neg, **s.kwargs())
    if np is None:
        raise NumpyException("Numpy is required for the warping_paths method")
    cost, result_fn, ival_fn = innerdistance.inner_dist_fns(s.inner_dist, use_ndim=s.use_ndim)
    r, c = len(s1), len(s2)
    if s.adj_max_length_diff is not None and abs(r - c) > s.adj_max_length_diff:
        return inf
    psi_1b, psi_1e, psi_2b, psi_2e = s.split_psi()
    dtw = np.full((r + 1, c + 1), inf)
    # dtw[0, 0] = 0
    for i in range(psi_2b + 1):
        dtw[0, i] = 0
    for i in range(psi_1b + 1):
        dtw[i, 0] = 0
    i1 = 0
    sc = 0
    ec = 0
    for i in range(r):
        i0 = i
        i1 = i + 1
        j_start = max(0, i - max(0, r - c) - s.window + 1)
        j_end = min(c, i + max(0, c - r) + s.window)
        if sc > j_start:
            j_start = sc
        smaller_found = False
        ec_next = i
        for j in range(j_start, j_end):
            d = cost(s1[i], s2[j])
            if s.adj_max_step is not None and d > s.adj_max_step:
                continue
            dtw[i1, j + 1] = d + min(dtw[i0, j],
                                     dtw[i0, j + 1] + s.adj_penalty,
                                     dtw[i1, j] + s.adj_penalty)
            if dtw[i1, j + 1] > s.adj_max_dist:
                if not smaller_found:
                    sc = j + 1
                if j >= ec:
                    break
            else:
                smaller_found = True
                ec_next = j + 1
        ec = ec_next
    # Decide which d to return
    if not keep_int_repr:
        dtw = result_fn(dtw)
    if psi_1e == 0 and psi_2e == 0:
        d = dtw[i1, min(c, c + s.window - 1)]
    else:
        ir = i1
        ic = min(c, c + s.window - 1)
        if psi_1e != 0:
            vr = dtw[ir:max(0, ir-psi_1e-1):-1, ic]
            mir = argmin(vr)
            vr_mir = vr[mir]
        else:
            mir = ir
            vr_mir = inf
        if psi_2e != 0:
            vc = dtw[ir, ic:max(0, ic-psi_2e-1):-1]
            mic = argmin(vc)
            vc_mic = vc[mic]
        else:
            mic = ic
            vc_mic = inf
        if vr_mir < vc_mic:
            if psi_neg:
                dtw[ir:ir-mir:-1, ic] = -1
            d = vr_mir
        else:
            if psi_neg:
                dtw[ir, ic:ic-mic:-1] = -1
            d = vc_mic
    if keep_int_repr:
        if s.adj_max_dist and d > s.adj_max_dist:
            d = inf
    else:
        if s.max_dist and d > s.max_dist:
            d = inf
    return d, dtw


def warping_paths_fast(s1, s2, psi_neg=True, keep_int_repr=False, compact=False, **kwargs):
    """Fast C version of :meth:`warping_paths`.

    The `use_pruning` argument is still False by default in case one needs the
    full matrix. If this is not the case, then `use_pruning` can be set to True.

    :param s1: See :meth:`warping_paths`
    :param s2: See :meth:`warping_paths`
    :param psi_neg: See :meth:`warping_paths`
    :param compact: Return a compact warping paths matrix.
        Size is ((l1 + 1), min(l2 + 1, abs(l1 - l2) + 2*window + 1)).
        This option is meant for internal use. For more details, see the C code.
    :param kwargs: See :meth:`warping_paths`
    """
    s1 = util_numpy.verify_np_array(s1)
    s2 = util_numpy.verify_np_array(s2)
    r = len(s1)
    c = len(s2)
    _check_library(raise_exception=True)
    settings = DTWSettings.for_dtw(s1, s2, **kwargs)
    if compact:
        wps_width = dtw_cc.wps_width(r, c, **settings.c_kwargs())
        wps_compact = np.full((len(s1)+1, wps_width), inf)
        if settings.use_ndim:
            d = dtw_cc.warping_paths_compact_ndim(wps_compact, s1, s2, psi_neg, keep_int_repr, **settings.c_kwargs())
        else:
            d = dtw_cc.warping_paths_compact(wps_compact, s1, s2, psi_neg, keep_int_repr, **settings.c_kwargs())
        return d, wps_compact

    dtw = np.full((r + 1, c + 1), inf)
    if settings.use_ndim:
        d = dtw_cc.warping_paths_ndim(dtw, s1, s2, psi_neg, keep_int_repr, **settings.c_kwargs())
    else:
        d = dtw_cc.warping_paths(dtw, s1, s2, psi_neg, keep_int_repr, **settings.c_kwargs())
    return d, dtw


def warping_paths_affinity(s1, s2, window=None, only_triu=False,
                           penalty=None, psi=None, psi_neg=True,
                           gamma=1, tau=0, delta=0, delta_factor=1,
                           use_c=False):
    """
    Dynamic Time Warping paths using an affinity/similarity matrix instead of a distance matrix.

    The full matrix of all warping paths (or accumulated cost matrix) is built.

    :param s1: First sequence
    :param s2: Second sequence
    :param window: see :meth:`distance`
    :param only_triu: Only compute upper traingular matrix of warping paths.
        This is useful if s1 and s2 are the same series and the matrix would be mirrored around the diagonal.
    :param penalty: see :meth:`distance`
    :param psi: see :meth:`distance`
    :param psi_neg: Replace values that should be skipped because of psi-relaxation with -1.
    :param gamma:
    :param tau:
    :param delta:
    :param delta_factor:
    :param use_c:
    :returns: (DTW distance, DTW matrix)
    """
    if use_c:
        return warping_paths_affinity_fast(s1, s2, window=window, only_triu=only_triu,
                                           penalty=penalty, tau=tau, delta=delta, delta_factor=delta_factor)
    if np is None:
        raise NumpyException("Numpy is required for the warping_paths method")
    s = DTWSettings.for_dtw(s1, s2, window=window, psi=psi, penalty=penalty, use_c=use_c)
    r, c = len(s1), len(s2)
    psi_1b, psi_1e, psi_2b, psi_2e = s.split_psi()
    dtw = np.full((r + 1, c + 1), -inf)
    # dtw[0, 0] = 0
    for i in range(psi_2b + 1):
        dtw[0, i] = 0
    for i in range(psi_1b + 1):
        dtw[i, 0] = 0
    i1 = 0
    for i in range(r):
        i0 = i
        i1 = i + 1
        j_start = max(0, i - max(0, r - c) - s.window + 1)
        if only_triu:
            j_start = max(i, j_start)
        j_end = min(c, i + max(0, c - r) + s.window)
        for j in range(j_start, j_end):
            d = np.exp(-gamma*(s1[i] - s2[j])**2)
            # print(f"{s1[i] - s2[j]=} -> {d=}")
            dtw_prev = max(dtw[i0, j],
                           dtw[i0, j + 1] - s.penalty,
                           dtw[i1, j] - s.penalty)
            if d < tau:
                dtw[i1, j + 1] = max(0, delta + delta_factor * dtw_prev)
            else:
                dtw[i1, j + 1] = max(0, d + dtw_prev)

    # Decide which d to return
    if psi_1e == 0 and psi_2e == 0:
        d = dtw[i1, min(c, c + s.window - 1)]
    else:
        ir = i1
        ic = min(c, c + s.window - 1)
        if psi_1e != 0:
            vr = dtw[ir:max(0, ir-psi_1e-1):-1, ic]
            mir = argmax(vr)
            vr_mir = vr[mir]
        else:
            mir = ir
            vr_mir = inf
        if psi_2e != 0:
            vc = dtw[ir, ic:max(0, ic-psi_2e-1):-1]
            mic = argmax(vc)
            vc_mic = vc[mic]
        else:
            mic = ic
            vc_mic = inf
        if vr_mir > vc_mic:
            if psi_neg:
                dtw[ir:ir-mir:-1, ic] = -1
            d = vr_mir
        else:
            if psi_neg:
                dtw[ir, ic:ic-mic:-1] = -1
            d = vc_mic
    return d, dtw


def warping_paths_affinity_fast(s1, s2, window=None, only_triu=False,
                                penalty=None, psi=None, psi_neg=True,
                                gamma=1, tau=0, delta=0, delta_factor=1,
                                compact=False, use_ndim=False):
    """Fast C version of :meth:`warping_paths`.

    :param s1:
    :param s2:
    :param window:
    :param only_triu:
    :param penalty:
    :param psi:
    :param psi_neg:
    :param gamma:
    :param tau:
    :param delta:
    :param delta_factor:
    :param compact: Return a compact warping paths matrix.
        Size is ((l1 + 1), min(l2 + 1, abs(l1 - l2) + 2*window + 1)).
        This option is meant for internal use. For more details, see the C code.
    :param use_ndim:
    """
    s1 = util_numpy.verify_np_array(s1)
    s2 = util_numpy.verify_np_array(s2)
    r = len(s1)
    c = len(s2)
    _check_library(raise_exception=True)
    settings = DTWSettings.for_dtw(s1, s2, window=window, penalty=penalty, psi=psi)
    if compact:
        wps_width = dtw_cc.wps_width(r, c, **settings.c_kwargs())
        wps_compact = np.full((len(s1)+1, wps_width), -inf)
        if use_ndim:
            d = dtw_cc.warping_paths_compact_ndim_affinity(wps_compact, s1, s2, only_triu,
                                                           gamma, tau, delta, delta_factor, psi_neg,
                                                           **settings.c_kwargs())
        else:
            d = dtw_cc.warping_paths_compact_affinity(wps_compact, s1, s2, only_triu,
                                                      gamma, tau, delta, delta_factor, psi_neg, **settings.c_kwargs())
        return d, wps_compact

    dtw = np.full((r + 1, c + 1), -inf)
    if use_ndim:
        d = dtw_cc.warping_paths_affinity_ndim(dtw, s1, s2, only_triu,
                                               gamma, tau, delta, delta_factor, psi_neg, **settings.c_kwargs())
    else:
        d = dtw_cc.warping_paths_affinity(dtw, s1, s2, only_triu,
                                          gamma, tau, delta, delta_factor, psi_neg, **settings.c_kwargs())
    return d, dtw


def distance_matrix_func(use_c=False, parallel=False, show_progress=False):
    def distance_matrix_wrapper(seqs, **kwargs):
        return distance_matrix(seqs, parallel=parallel, use_c=use_c,
                               show_progress=show_progress, **kwargs)
    return distance_matrix_wrapper


def distance_matrix(s, block=None, compact=False, parallel=False,
                    use_mp=False, show_progress=False, only_triu=False, **kwargs):
    """Distance matrix for all sequences in s.

    :param s: Iterable of series
    :param block: Only compute block in matrix. Expects tuple with begin and end, e.g. ((0,10),(20,25)) will
        only compare rows 0:10 with rows 20:25.
    :param compact: Return the distance matrix as an array representing the upper triangular matrix.
    :param parallel: Use parallel operations
    :param use_mp: Force use Multiprocessing for parallel operations (not OpenMP)
    :param show_progress: Show progress using the tqdm library. This is only supported for
        the pure Python version (thus not the C-based implementations).
    :param only_triu: Only compute upper traingular matrix of warping paths.
        This is useful if s1 and s2 are the same series and the matrix would be mirrored around the diagonal.
    :param kwargs: See arguments for :class:`DTWSettings`
    :returns: The distance matrix or the condensed distance matrix if the compact argument is true
    """
    settings = DTWSettings(**kwargs)
    # Check whether multiprocessing is available
    if settings.use_c:
        requires_omp = parallel and not use_mp
        _check_library(raise_exception=True, include_omp=requires_omp)
    if parallel and (use_mp or not settings.use_c):
        try:
            import multiprocessing as mp
            logger.info('Using multiprocessing')
        except ImportError:
            msg = 'Cannot load multiprocessing'
            logger.error(msg)
            raise Exception(msg)
    else:
        mp = None
    if block is not None:
        if len(block) > 2 and block[2] is False and compact is False:
            raise Exception(f'Block cannot have a third argument triu=false with compact=false')
        if (block[0][1] - block[0][0]) < 1 or (block[1][1] - block[1][0]) < 1:
            return []
    # Prepare options and data to pass to distance method
    dist_opts = settings.kwargs()
    s = SeriesContainer.wrap(s)
    if settings.use_ndim:
        ndim = s.detected_ndim
    else:
        ndim = 1
    if settings.use_c:
        for k, v in dist_opts.items():
            if v is None:
                # None is represented as 0.0 for C
                dist_opts[k] = 0

    logger.info('Computing distances')
    if settings.use_c and parallel and not use_mp and dtw_cc_omp is not None:
        logger.info("Compute distances in C (parallel=OMP)")
        dist_opts['block'] = block
        if settings.use_ndim:
            dists = dtw_cc_omp.distance_matrix_ndim(s, ndim, **dist_opts)
        else:
            dists = dtw_cc_omp.distance_matrix(s, **dist_opts)

    elif settings.use_c and parallel and (dtw_cc_omp is None or use_mp):
        logger.info("Compute distances in C (parallel=MP)")
        idxs = _distance_matrix_idxs(block, len(s))
        if settings.use_ndim:
            fn = _distance_c_with_params_ndim
        else:
            fn = _distance_c_with_params
        with mp.Pool() as p:
            dists = p.map(fn, [(s[r], s[c], dist_opts) for c, r in zip(*idxs)])

    elif settings.use_c and not parallel:
        logger.info("Compute distances in C (parallel=No)")
        dist_opts['block'] = block
        if settings.use_ndim:
            dists = dtw_cc.distance_matrix_ndim(s, ndim, **dist_opts)
        else:
            dists = dtw_cc.distance_matrix(s, **dist_opts)

    elif not settings.use_c and parallel:
        logger.info("Compute distances in Python (parallel=MP)")
        idxs = _distance_matrix_idxs(block, len(s))
        if settings.use_ndim:
            fn = _distance_with_params_ndim
        else:
            fn = _distance_with_params
        with mp.Pool() as p:
            dists = p.map(fn, [(s[r], s[c], dist_opts) for c, r in zip(*idxs)])

    elif not settings.use_c and not parallel:
        logger.info("Compute distances in Python (parallel=No)")
        dists = distance_matrix_python(s, block=block, show_progress=show_progress, settings=settings)

    else:
        raise Exception(f'Unsupported combination of: parallel={parallel}, '
                        f'use_c={s.use_c}, dtw_cc_omp={dtw_cc_omp}, use_mp={use_mp}')

    exp_length = _distance_matrix_length(block, len(s))
    assert len(dists) == exp_length, "len(dists)={} != {} (block = {})".format(len(dists), exp_length, block)
    if compact:
        return dists

    # Create full matrix and fill upper triangular matrix with distance values (or only block if specified)
    dists_matrix = distances_array_to_matrix(dists, nb_series=len(s), block=block, only_triu=only_triu)

    return dists_matrix


def distances_array_to_matrix(dists, nb_series, block=None, only_triu=False):
    """Transform a condensed distances array to a full matrix representation.

    The upper triangular matrix will contain all the distances.
    """
    if np is None:
        raise NumpyException("Numpy is required for the distances_array_to_matrix method, "
                             "set compact to true")
    dists_matrix = np.full((nb_series, nb_series), inf, dtype=DTYPE)
    idxs = _distance_matrix_idxs(block, nb_series)
    dists_matrix[idxs] = dists
    if not only_triu:
        dists_matrix.T[idxs] = dists
        np.fill_diagonal(dists_matrix, 0)
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


def distance_matrix_python(s, block=None, show_progress=False, settings=None):
    if settings is None:
        settings = DTWSettings()
    dists = array.array('d', [inf] * _distance_matrix_length(block, len(s)))
    block, triu = _complete_block(block, len(s))
    it_r = range(block[0][0], block[0][1])
    if show_progress:
        it_r = tqdm(it_r)
    idx = 0
    for r in it_r:
        if triu:
            it_c = range(max(r + 1, block[1][0]), min(len(s), block[1][1]))
        else:
            it_c = range(block[1][0], min(len(s), block[1][1]))
        for c in it_c:
            dists[idx] = distance(s[r], s[c], **settings.kwargs())
            idx += 1
    return dists


def _complete_block(block, nb_series):
    """Expand block variable to represent exact indices of ranges.

    :param block: None, 0, or tuple
    :param nb_series: Number of series in the list
    :return: Block with filled in indices, Boolean to indicate triu
    """
    if block is None or block == 0:
        block = ((0, nb_series), (0, nb_series))
        return block, True
    else:
        if len(block) > 2 and block[2] is False:
            return block, False
        else:
            return block, True


def _distance_matrix_idxs(block, nb_series):
    if block is None or block == 0:
        if np is not None:
            idxs = np.triu_indices(nb_series, k=1)
            return idxs
    # Numpy is not available or not triu
    block, triu = _complete_block(block, nb_series)
    idxsl_r = []
    idxsl_c = []
    for r in range(block[0][0], block[0][1]):
        if triu:
            it_c = range(max(r + 1, block[1][0]), min(nb_series, block[1][1]))
        else:
            it_c = range(block[1][0], min(nb_series, block[1][1]))
        for c in it_c:
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
        if len(block) > 2 and block[2] is False:
            length = (block_re - block_rb) * (block_ce - block_cb)
        else:
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


def distance_matrix_fast(s, max_dist=None, use_pruning=True, max_length_diff=None,
                         window=None, max_step=None, penalty=None, psi=None,
                         block=None, compact=False, parallel=True, use_mp=False,
                         only_triu=False, inner_dist=innerdistance.default, use_c=True):
    """Same as :meth:`distance_matrix` but with different defaults to choose the
    fast parallized C version (use_c = True, parallel = True, use_pruning = True).

    This method uses the C-compiled version of the DTW algorithm and uses parallelization.
    By default, this is the OMP C parallelization. If the OMP functionality is not available
    the parallelization is changed to use Python's multiprocessing library.
    """
    _check_library(raise_exception=True, include_omp=False)
    if not use_mp and parallel:
        try:
            _check_library(raise_exception=True, include_omp=True)
        except CythonException:
            use_mp = True
    return distance_matrix(s, max_dist=max_dist, use_pruning=use_pruning,
                           max_length_diff=max_length_diff, window=window,
                           max_step=max_step, penalty=penalty, psi=psi,
                           block=block, compact=compact, parallel=parallel,
                           use_c=True, use_mp=use_mp, show_progress=False,
                           only_triu=only_triu, inner_dist=inner_dist)


def warping_path(from_s, to_s, include_distance=False, use_ndim=False, **kwargs):
    """Compute warping path between two sequences."""
    dist, paths = warping_paths(from_s, to_s, use_ndim=use_ndim, **kwargs)
    path = best_path(paths)
    if include_distance:
        return path, dist
    return path


def warping_path_fast(from_s, to_s, include_distance=False, **kwargs):
    """Compute warping path between two sequences."""
    from_s, to_s, settings_kwargs = warping_path_args_to_c(from_s, to_s, **kwargs)
    result = dtw_cc.warping_path(from_s, to_s, include_distance=include_distance,
                                 **settings_kwargs)
    return result


def warping_path_prob(from_s, to_s, avg, include_distance=False, use_c=True, **kwargs):
    """Compute warping path between two sequences."""
    if not use_c:
        raise AttributeError('warping_path_prob with use_c=False not yet supported')
    from_s, to_s, settings_kwargs = warping_path_args_to_c(from_s, to_s, **kwargs)
    result = dtw_cc.warping_path_prob(from_s, to_s, avg,
                                      include_distance=include_distance, **settings_kwargs)
    return result


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

    :returns DTW distance, the best path, DTW distance between 2 path elements, DTW matrix
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


def best_path(paths, row=None, col=None, use_max=False, penalty=0):
    """Compute the optimal path from the nxm warping paths matrix.

    :param paths: Warping paths matrix
    :param row: If given, start from this row (instead of lower-right corner)
    :param col: If given, start from this column (instead of lower-right corner)
    :param use_max: Find maximal path instead of minimal path
    :param penalty: The penalty used.
        If this is used, then paths should be expressed as the internal representation.
    :return: Array of (row, col) representing the best path
    """
    if use_max:
        argm = argmax
    else:
        argm = argmin
    if row is None:
        i = int(paths.shape[0] - 1)
    else:
        i = row
    if col is None:
        j = int(paths.shape[1] - 1)
    else:
        j = col
    p = []
    if paths[i, j] != -1:
        p.append((i - 1, j - 1))
    while i > 0 and j > 0:
        c = argm([paths[i - 1, j - 1],
                  paths[i - 1, j] + penalty,
                  paths[i, j - 1] + penalty])
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
    if v != -1:
        path.append((r - 1, c - 1))
    while r > 0 and c > 0:
        if v == -1:
            v = np.Inf
        r_c, c_c = r, c
        if r >= 1 and c >= 1 and m[r - 1, c - 1] <= v:
            r_c, c_c, v = r - 1, c - 1, m[r - 1, c - 1]
        if r >= 1 and m[r - 1, c] <= v:
            r_c, c_c, v = r - 1, c, m[r - 1, c]
        if c >= 1 and m[r, c - 1] <= v:
            r_c, c_c, v = r, c - 1, m[r, c - 1]
        if v != -1:
            path.append((r_c - 1, c_c - 1))
        r, c = r_c, c_c
    path.pop()
    path.reverse()
    return path


def warping_path_args_to_c(s1, s2, **kwargs):
    s1 = util_numpy.verify_np_array(s1)
    s2 = util_numpy.verify_np_array(s2)
    _check_library(raise_exception=True)

    def get(key):
        value = kwargs.get(key, None)
        if value is None:
            return 0
        return value
    settings_kwargs = {key: get(key) for key in
                       ['window', 'max_dist', 'max_step', 'max_length_diff', 'penalty', 'psi']}
    return s1, s2, settings_kwargs
