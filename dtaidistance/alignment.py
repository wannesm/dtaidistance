# -*- coding: UTF-8 -*-
"""
dtaidistance.alignment
~~~~~~~~~~~~~~~~~~~~~~

Sequence alignment (e.g. Needleman–Wunsch).

:author: Wannes Meert
:copyright: Copyright 2017-2018 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import math
import numpy as np

from .dtw import dp


def _needleman_wunsch_fn(v1, v2):
    """Needleman-Wunsch

    Match: +1 -> -1
    Mismatch or Indel: −1 -> +1

    The values are reversed because our general dynamic programming algorithm
    selects the minimal value instead of the maximal value.
    """
    d_indel = 1  # gap / indel
    if v1 == v2:
        d = -1  # match
    else:
        d = 1  # mismatch
    return d, d_indel


def _needleman_wunsch_border(ri, ci):
    if ri == 0:
        return ci
    if ci == 0:
        return ri
    return 0


def needleman_wunsch(s1, s2, window=None, max_dist=None,
                     max_step=None, max_length_diff=None, psi=None):
    """Needleman-Wunsch global sequence alignment."""
    value, matrix = dp(s1, s2,
                       _needleman_wunsch_fn, border=_needleman_wunsch_border,
                       penalty=0, window=window, max_dist=max_dist,
                       max_step=max_step, max_length_diff=max_length_diff, psi=psi)
    matrix = -matrix
    return value, matrix
