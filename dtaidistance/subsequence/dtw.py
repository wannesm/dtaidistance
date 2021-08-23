# -*- coding: UTF-8 -*-
"""
dtaidistance.subsequence.dtw
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(requires version 2.3.0 or higher)

DTW-based subsequence matching.

:author: Wannes Meert
:copyright: Copyright 2021 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import numpy.ma as ma

from .. import dtw # import warping_paths, warping_paths_fast, best_path, warping_paths_affinity, distance
from .. import dtw_ndim
from .. import util_numpy
from .. import util


try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
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


logger = logging.getLogger("be.kuleuven.dtai.distance")


dtw_cc = None
try:
    from . import dtw_cc
except ImportError:
    dtw_cc = None


def subsequence_alignment(query, series, use_c=False):
    """See SubsequenceAligment.

    :param query:
    :param series:
    :return:
    """
    sa = SubsequenceAlignment(query, series, use_c=use_c)
    sa.align()
    return sa


class SAMatch:
    def __init__(self, idx, alignment):
        """SubsequenceAlignment match"""
        self.idx = idx
        self.alignment = alignment

    @property
    def value(self):
        return self.alignment.matching[self.idx]

    @property
    def segment(self):
        """Matched segment in series."""
        start = self.alignment.matching_function_startpoint(self.idx)
        end = self.alignment.matching_function_endpoint(self.idx)
        return [start, end]

    @property
    def path(self):
        """Matched path in series"""
        return self.alignment.matching_function_bestpath(self.idx)

    def __str__(self):
        return f'SAMatch({self.idx})'

    def __repr__(self):
        return self.__str__()


class SubsequenceAlignment:
    def __init__(self, query, series, penalty=0.1, use_c=False):
        """Subsequence alignment using DTW.
        Find where the query occurs in the series.

        Based on Fundamentals of Music Processing, Meinard Müller, Springer, 2015.

        Example:
        query = np.array([1., 2, 0])
        series = np.array([1., 0, 1, 2, 1, 0, 2, 0, 3, 0, 0])
        sa = subsequence_search(query, series)
        mf = sa.matching_function()
        sa.kbest_matches(k=2)


        :param query: Subsequence to search for
        :param series: Long sequence in which to search
        :param penalty: Penalty for non-diagonal matching
        :param use_c: Use the C-based DTW function if available
        """
        self.query = query
        self.series = series
        self.penalty = penalty
        self.paths = None
        self.matching = None
        self.use_c = use_c

    def reset(self):
        self.matching = None

    def align(self):
        if self.matching is not None:
            return
        psi = [0, 0, len(self.series), len(self.series)]
        if not self.use_c:
            _, self.paths = dtw.warping_paths(self.query, self.series, penalty=self.penalty, psi=psi,
                                              psi_neg=False)
        else:
            _, self.paths = dtw.warping_paths_fast(self.query, self.series, penalty=self.penalty, psi=psi,
                                                   compact=False, psi_neg=False)
        self._compute_matching()

    def align_fast(self):
        self.use_c = True
        return self.align()

    def _compute_matching(self):
        matching = self.paths[-1, :]
        if len(matching) > len(self.series):
            matching = matching[-len(self.series):]
        self.matching = np.array(matching) / len(self.query)

    def warping_paths(self):
        """Get matrix with all warping paths.

        If the aligmnent was computed using a compact, the paths are first copied into a full
        warping paths matrix.

        :return: Numpy matrix of size (len(query)+1) * (len(series)+1)
        """
        return self.paths

    def matching_function(self):
        """The matching score for each end-point of a possible match."""
        return self.matching

    def get_match(self, idx):
        return SAMatch(idx, self)

    def best_match(self):
        best_idx = np.argmin(self.matching)
        return self.get_match(best_idx)

    def kbest_matches(self, k=1, overlap=0):
        """Yields the next best match. Stops at k matches (use None for all matches).

        :param k: Number of matches to yield. None is all matches.
        :param overlap: Matches cannot overlap unless overlap > 0.
        :return: Yield an SAMatch object
        """
        self.align()
        matching = np.array(self.matching)
        maxv = np.ceil(np.max(matching) + 1)
        matching[:min(len(self.query) - 1, overlap)] = maxv
        ki = 0
        while k is None or ki < k:
            best_idx = np.argmin(matching)
            if best_idx == 0 or np.isinf(matching[best_idx]) or matching[best_idx] == maxv:
                # No more matches found
                break
            match = self.get_match(best_idx)
            b, e = match.segment
            cur_overlap = min(overlap, e - b - 1)
            mb, me = best_idx + 1 - (e - b) + cur_overlap, best_idx + 1
            if np.isinf(np.max(matching[mb:me])):
                # No overlapping matches
                matching[best_idx] = maxv
                continue
            matching[mb:me] = np.inf
            ki += 1
            yield match

    def matching_function_segment(self, idx):
        """Matched segment in series."""
        start = self.matching_function_startpoint(idx)
        end = self.matching_function_endpoint(idx)
        return [start, end]

    def matching_function_endpoint(self, idx):
        """Index in series for end of match in matching function at idx.

        :param idx: Index in matching function
        :return: Index in series
        """
        if len(self.matching) == len(self.series):
            return idx
        diff = len(self.series) - len(self.matching)
        return idx + diff

    def matching_function_startpoint(self, idx):
        """Index in series for start of match in matching function at idx.

        :param idx: Index in matching function
        :return: Index in series
        """
        real_idx = idx + 1
        path = dtw.best_path(self.paths, col=real_idx)
        start_idx = path[0][1]
        return start_idx

    def matching_function_bestpath(self, idx):
        """Indices in series for best path for match in matching function at idx.

        :param idx: Index in matching function
        :return: List of (row, col)
        """
        real_idx = idx + 1
        path = dtw.best_path(self.paths, col=real_idx)
        return path


def local_concurrences(series1, series2=None, gamma=1, tau=0, delta=0, delta_factor=1, estimate_settings=None,
                       only_triu=False, penalty=None):
    """Local concurrences, see LocalConcurrences.

    :param series1:
    :param series2:
    :param gamma: Affinity transformation exp(-gamma*(s1[i] - s2[j])**2)
    :param tau: threshold parameter
    :param delta: penalty parameter
        Should be negative. Added instead of the affinity score (if score below tau threshold parameter).
    :param delta_factor: multiply cumulative score (e.g. by 0.5).
        This is useful to have the same impact at different locations in the warping paths matrix, which
        is cumulative (and thus typically large in one corner and small in the opposite corner).
    :param estimate_settings: Estimate tau, delta, delta_factor from given series. Will be passed as
        tau_std to estimate_settings_from_std.
    :param only_triu: Only compute the upper traingle matrix values. Useful to avoid redundant computations
        when series1 is equal to series2 (or equivalently if series2 is None).
    :param penalty: Penalty that is added when dynamic programming is using moving vertically or horizontally
        through the matrix instead of diagonally. Used to prefer diagonal paths.
    :return:
    """
    lc = LocalConcurrences(series1, series2, gamma, tau, delta, delta_factor,
                           only_triu=only_triu, penalty=penalty)
    if estimate_settings is not None:
        lc.estimate_settings_from_std(series1, estimate_settings)
    lc.align()
    return lc


class LCMatch:
    def __init__(self, lc, row=None, col=None):
        """LocalConcurrences match"""
        self.row = row  # type: int
        self.col = col  # type: int
        self.lc = lc  # type: LocalConcurrences
        self._path = None

    @property
    def path(self):
        if self._path is not None:
            return self._path
        self._path = self.lc.best_path(self.row, self.col)
        return self._path

    def __str__(self):
        return f'LCMatch({self.row, self.col})'

    def __repr__(self):
        return self.__str__()


class LocalConcurrences:
    def __init__(self, series1, series2=None, gamma=1, tau=0, delta=0, delta_factor=1, only_triu=False, penalty=None):
        """Version identification based on local concurrences.

        Find recurring patterns across two time series. Used to identify whether one time series is
        a version of another. If the two time series are the some one, it can be used to find typical
        or frequent patterns in a time series.

        Based on 7.3.2 Identiﬁcation Procedure in Fundamentals of Music Processing, Meinard Müller, Springer, 2015.

        Different from the original formulation, D_tau is introduced based on the given delta factor.
        This makes the penalty less sensitive to the cumulative effect of the paths in the
        self-similarity matrix S:

        S_tau(n,m) = S(n,m)  if  S(n,m) >= tau  (with tau >= 0)
                     delta   if  S(n,m) < tau   (with delta <= 0)

        And for the accumulated score matrix D:

        D_tau(n,m) = max(0,
                         df * D_tau(n−1,m−1) + S_tau(n,m),
                         df * D_tau(n−1,m)   + S_tau(n,m),
                         df * D_tau(n,m−1)   + S_tau(n,m))
        where df = 1 if S(n,m) >= tau and df=delta_factor (<=1) otherwise,

        :param series1: First time series.
        :param series2: Second time series. If empty, series1 is used and compared with itself.
        :param gamma: Affinity transformation exp(-gamma*(s1[i] - s2[j])**2), should be >0
        :param tau: threshold parameter, should be >= 0
        :param delta: penalty parameter, should be <= 0
        :param delta_factor: penalty factor parameter, should be <= 1
        :param only_triu: Only consider upper triangular matrix in warping paths.
        """
        self.series1 = series1
        if series2 is None:
            # Self-comparison
            self.series2 = self.series1
            self.only_triu = True
        else:
            self.series2 = series2
            if len(series1) == len(series2):
                self.only_triu = only_triu
            else:
                self.only_triu = False
        self.gamma = gamma
        self.tau = tau
        self.delta = delta
        self.delta_factor = delta_factor
        self.penalty = penalty
        self._wp = None  # warping paths

    def reset(self):
        self._wp = None

    def estimate_settings_from_std(self, series, tau_std=0.33):
        """

        :param series:
        :param tau_std: Set tau to differences larger than tau_std time standard deviation of
            the given series (default is 0.33, or reject differences that are larger than
            the deviation wrt to the mean of 75% of the values in the series, assuming a
            normal distribution).
        :return:
        """
        diffp = tau_std * np.std(series)
        self.delta = -2 * np.exp(-self.gamma * diffp ** 2)
        self.delta_factor = 0.5
        self.tau = np.exp(-self.gamma * diffp ** 2)

    def align(self):
        """

        :return:
        """
        if self._wp is not None:
            return
        _, wp = dtw.warping_paths_affinity(self.series1, self.series2,
                                           gamma=self.gamma, tau=self.tau,
                                           delta=self.delta, delta_factor=self.delta_factor,
                                           only_triu=self.only_triu, penalty=self.penalty)
        self._wp = ma.masked_array(wp)
        if self.only_triu:
            il = np.tril_indices(self._wp.shape[0])
            self._wp[il] = ma.masked

    @property
    def wp(self):
        return self._wp.data

    def best_match(self):
        idx = np.unravel_index(np.argmax(self._wp, axis=None), self._wp.shape)
        r, c = idx
        lcm = LCMatch(self, r, c)
        # path = lcm.path
        # for (x, y) in path:
        #     self._wp[x + 1, y + 1] = ma.masked
        return lcm

    def kbest_matches(self, k=1, minlen=2, buffer=0):
        """Yields the next best match. Stops at k matches (use None for all matches).

        :param k: Number of matches to yield. None is all matches.
        :param minlen: Consider only matches of length longer than minlen
        :param buffer: Matches cannot be closer than buffer to each other.
        :return: Yield an LCMatch object
        """
        ki = 0
        while k is None or ki < k:
            idx = None
            lcm = None
            while idx is None:
                idx = np.unravel_index(np.argmax(self._wp, axis=None), self._wp.shape)
                if idx[0] == 0 or idx[1] == 0:
                    return None
                r, c = idx
                lcm = LCMatch(self, r, c)
                for (x, y) in lcm.path:
                    x += 1
                    y += 1
                    if len(self._wp.mask.shape) > 0 and self._wp.mask[x, y] is True:  # True means invalid
                        # print('found path contains masked, restart')
                        lcm = None
                        idx = None
                        break
                    else:
                        self._wp[x, y] = ma.masked
                if len(lcm.path) < minlen:
                    # print('found path too short, restart')
                    lcm = None
                    idx = None
            if buffer > 0 and lcm is not None:
                miny, maxy = 0, self._wp.shape[1] - 1
                minx, maxx = 0, self._wp.shape[0] - 1
                for (x, y) in lcm.path:
                    xx = x + 1
                    for yy in range(max(miny, y + 1 - buffer), min(maxy, y + 1 + buffer)):
                        self._wp[xx, yy] = ma.masked
                    yy = y + 1
                    for xx in range(max(minx, x + 1 - buffer), min(maxx, x + 1 + buffer)):
                        self._wp[xx, yy] = ma.masked
            if lcm is not None:
                ki += 1
                yield lcm

    def best_path(self, row, col):
        if self._wp is None:
            return None
        argm = argmax
        i = row
        j = col
        p = [(i - 1, j - 1)]
        # prev = self._wp[i, j]
        while i > 0 and j > 0:
            values = [self._wp.data[i - 1, j - 1], self._wp.data[i - 1, j], self._wp.data[i, j - 1]]
            # print(f'{i=}, {j=}, {argm(values)=}, {ma.argmax(values)=}, {values=}')
            c = argm(values)
            # if values[c] is ma.masked:
            #     break
            if values[c] <= 0:  # values[c] > prev:
                break
            # prev = values[c]
            if c == 0:
                if self._wp[i - 1, j - 1] is ma.masked:
                    break
                i, j = i - 1, j - 1
            elif c == 1:
                if self._wp[i - 1, j] is ma.masked:
                    break
                i = i - 1
            elif c == 2:
                if self._wp[i, j - 1] is ma.masked:
                    break
                j = j - 1
            p.append((i - 1, j - 1))
        if p[-1][0] < 0 or p[-1][1] < 0:
            p.pop()
        p.reverse()
        return p


def subsequence_search(query, series):
    """See SubsequenceSearch.

    :param query:
    :param series:
    :return:
    """
    ss = SubsequenceSearch(query, series)
    return ss


class SSMatch:
    def __init__(self, idx, ss):
        self.idx = idx
        self.ss = ss

    def __str__(self):
        return f'SSMatch({self.idx})'

    def __repr__(self):
        return self.__str__()


class SubsequenceSearch:
    def __init__(self, query, s, dists_options=None):
        """Search the best matching (subsequence) time series compared to a given time series.

        :param query: Time series to search for
        :param s: Iterator over time series to perform search on.
            This can be for example windows over a long time series.
        :param dists_options: Options for DTW
        """
        self.query = query
        self.s = s
        self.distances = None
        self.k = None
        self.dists_options = {} if dists_options is None else dists_options

    def reset(self):
        self.distances = None

    def align(self, k=None):
        if self.distances is not None and self.k >= k:
            return
        self.distances = np.zeros((len(self.s),))
        import heapq
        h = [-np.inf]
        max_dist = np.inf
        for idx, series in enumerate(self.s):
            dist = dtw.distance(self.query, series, **self.dists_options)
            if k is not None:
                if len(h) < k:
                    if not np.isinf(dist):
                        heapq.heappush(h, -dist)
                        max_dist = -min(h)
                else:
                    if not np.isinf(dist):
                        heapq.heappushpop(h, -dist)
                        max_dist = -min(h)
                self.dists_options['max_dist'] = max_dist
            self.distances[idx] = dist

    def best_match(self):
        self.align(k=1)
        best_idx = np.argmin(self.distances)
        return SSMatch(best_idx, self)

    def kbest_matches(self, k=1):
        self.align(k=k)
        if k is None:
            return [SSMatch(best_idx, self) for best_idx in range(len(self.distances))]
        best_idxs = np.argpartition(self.distances, k)
        return [SSMatch(best_idx, self) for best_idx in best_idxs[:k]]
