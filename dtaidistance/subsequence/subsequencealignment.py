# -*- coding: UTF-8 -*-
"""
dtaidistance.subsequence.subsequencealignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(requires version 2.3.0 or higher)

DTW-based subsequence matching.

:author: Wannes Meert
:copyright: Copyright 2021-2023 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging

from .. import dtw  # import warping_paths, warping_paths_fast, best_path, warping_paths_affinity, distance
from .. import dtw_ndim
from .. import util_numpy
from .. import util


try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
    import numpy.ma as ma
    argmin = np.argmin
    argmax = np.argmax
    array_min = np.min
    array_max = np.max
except ImportError:
    np = None
    ma = None
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
        """Normalized DTW distance of match.

        Normalization is the DTW distance divided by the query length.
        """
        return self.alignment.matching[self.idx]

    @property
    def distance(self):
        """DTW distance of match.

        This value is dependent on the length of the query. Use the value
        property when comparing queries of different lengths.
        """
        return self.value * len(self.alignment.query)

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

        Example::

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
        if np is not None and isinstance(self.series, np.ndarray) and len(self.series.shape) > 1:
            if not self.use_c:
                _, self.paths = dtw_ndim.warping_paths(self.query, self.series, penalty=self.penalty, psi=psi,
                                                       psi_neg=False)
            else:
                _, self.paths = dtw_ndim.warping_paths_fast(self.query, self.series, penalty=self.penalty, psi=psi,
                                                            compact=False, psi_neg=False)
        else:
            if not self.use_c:
                _, self.paths = dtw.warping_paths(self.query, self.series, penalty=self.penalty, psi=psi,
                                                  psi_neg=False)
            else:
                _, self.paths = dtw.warping_paths_fast(self.query, self.series, penalty=self.penalty, psi=psi,
                                                       compact=False, psi_neg=False)
        self._compute_matching()

    def align_fast(self):
        use_c = self.use_c
        self.use_c = True
        result = self.align()
        self.use_c = use_c
        return result

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

    def best_match_fast(self):
        use_c = self.use_c
        self.use_c = True
        result = self.best_match()
        self.use_c = use_c
        return result

    def best_match(self):
        best_idx = np.argmin(self.matching)
        return self.get_match(best_idx)

    def kbest_matches_fast(self, k=1, overlap=0):
        use_c = self.use_c
        self.use_c = True
        result = self.kbest_matches(k=k, overlap=overlap)
        self.use_c = use_c
        return result

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
