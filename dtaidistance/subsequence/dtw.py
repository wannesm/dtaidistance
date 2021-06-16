# -*- coding: UTF-8 -*-
"""
dtaidistance.subsequence.dtw
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DTW-based subsequence matching

:author: Wannes Meert
:copyright: Copyright 2021 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import numpy as np

from ..dtw import warping_paths, warping_paths_fast, best_path, _check_library


logger = logging.getLogger("be.kuleuven.dtai.distance")


dtw_cc = None
try:
    from . import dtw_cc
except ImportError:
    dtw_cc = None


def subsequence_search(query, series):
    sa = SubsequenceAlignment(query, series)
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
    def __init__(self, query, series, penalty=0.1):
        """Subsequence alignment using DTW.
        Find where the query occurs in the series.

        Based on Fundamentals of Music Processing, Meinard Müller, Springer, 2015.

        Example:
        query = np.array([1., 2, 0])
        series = np.array([1., 0, 1, 2, 1, 0, 2, 0, 3, 0, 0])
        sa = subsequence_search(query, series)
        mf = sa.matching_function()
        best_match_end_idx = np.argmin(mf)
        best_match_start_idx = sa.matching_function_startpoint(best_match_end_idx)
        best_match_path = sa.matching_function_bestpath(best_match_end_idx)


        :param query: Subsequence to search for
        :param series: Long sequence in which to search
        :param penalty: Penalty for non-diagonal matching
        """
        self.query = query
        self.series = series
        self.penalty = penalty
        self.paths = None
        self.matching = None

    def align(self, use_c=False):
        psi = [0, 0, len(self.series), len(self.series)]
        if use_c:
            _, self.paths = warping_paths(self.query, self.series, penalty=self.penalty, psi=psi,
                                          psi_neg=False)
        else:
            _, self.paths = warping_paths_fast(self.query, self.series, penalty=self.penalty, psi=psi,
                                               compact=False, psi_neg=False)
        self._compute_matching()

    def align_fast(self):
        return self.align(use_c=True)

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

    def kbest_match(self, k=1):
        # TODO remove overlapping matches
        best_idxs = np.argpartition(self.matching, kth=k)[:k]
        best_idxs = best_idxs[np.argsort(self.matching[best_idxs])]
        return [self.get_match(best_idx) for best_idx in best_idxs]

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
        path = best_path(self.paths, col=real_idx)
        start_idx = path[0][1]
        return start_idx

    def matching_function_bestpath(self, idx):
        """Indices in series for best path for match in matching function at idx.

        :param idx: Index in matching function
        :return: List of (row, col)
        """
        real_idx = idx + 1
        path = best_path(self.paths, col=real_idx)
        return path
