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

from ..dtw import warping_paths, warping_paths_fast


def subsequence_search(query, series):
    sa = SubsequenceAlignment(query, series)
    sa.align()
    return sa


class SubsequenceAlignment:
    def __init__(self, query, series, penalty=0.1):
        """Subsequence alignment using DTW.

        Based on Fundamentals of Music Processing, Meinard Müller, Springer, 2015.

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

    @property
    def matching_function(self):
        return self.matching


    def align_fast(self):
        return self.align(use_c=True)



