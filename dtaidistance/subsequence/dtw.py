# -*- coding: UTF-8 -*-
"""
dtaidistance.subsequence.dtw
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DTW-based subsequence matching

:author: Wannes Meert
:copyright: Copyright 2017-2018 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import numpy as np

from ..dtw import warping_paths, warping_paths_fast, best_path


def subsequence_search(query, series):
    sa = SubsequenceAlignment(query, series)
    sa.align()
    return sa


class SubsequenceAlignment:
    def __init__(self, query, series, penalty=0.1):
        """Subsequence alignment using DTW.

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
                                               compact=True, psi_neg=False)
        matching = self.paths[-1, :]
        if len(matching) > len(self.series):
            matching = matching[-len(self.series):]
        self.matching = np.array(matching) / len(self.query)

    def matching_function(self):
        """The matching score for each end-point of a possible match."""
        return self.matching

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

    def align_fast(self):
        return self.align(use_c=True)
