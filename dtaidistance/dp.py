# -*- coding: UTF-8 -*-
"""
dtaidistance.dp
~~~~~~~~~~~~~~~

Generic Dynamic Programming functions

:author: Wannes Meert
:copyright: Copyright 2017-2022 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
from enum import Enum
import logging

from .util_numpy import NumpyException

try:
    import numpy as np
except ImportError:
    np = None


logger = logging.getLogger("be.kuleuven.dtai.distance")

class Direction(Enum):
    """Define Unicode arrows we'll use in the traceback matrix."""
    UP = "\u2191"
    RIGHT = "\u2192"
    DOWN = "\u2193"
    LEFT = "\u2190"
    DOWN_RIGHT = "\u2198"
    UP_LEFT = "\u2196"


def dp(s1, s2, fn, border=None, window=None, max_dist=None,
       max_step=None, max_length_diff=None, penalty=None, psi=None):
    """
    Generic dynamic programming.

    This function does not optimize storage when a window size is given (e.g. in contrast with
    the fast DTW functions).

    :param s1: First sequence
    :param s2: Second sequence
    :param fn: Function to compare two items from both sequences and return the value to add to the current
        warping path (e.g. for Dynamic Time Warping this is (s1[i]-s2[j])**2,
        and for Needleman-Wunsch this is -1 for a match and +1 for a mismatch).
    :param border: Callable object to fill in the initial borders (border(row_idx, col_idx).
    :param window: see :meth:`distance`
    :param max_dist: see :meth:`distance`
    :param max_step: see :meth:`distance`
    :param max_length_diff: see :meth:`distance`
    :param penalty: see :meth:`distance`
    :param psi: see :meth:`distance`
    :returns: (cost, score matrix, paths matrix)
    """
    if np is None:
        raise NumpyException('Function dp requires Numpy.')
    r, c = len(s1), len(s2)
    # Set default parameters
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        return np.inf
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = np.inf
    if not max_dist:
        max_dist = np.inf
    if not penalty:
        penalty = 0
    if psi is None:
        psi = 0
    # Init scoring matrix
    scores = np.full((r + 1, c + 1), np.inf)
    if border:
        for ci in range(c + 1):
            scores[0, ci] = border(0, ci)
        for ri in range(1, r + 1):
            scores[ri, 0] = border(ri, 0)
    for i in range(psi + 1):
        scores[0, i] = 0
        scores[i, 0] = 0

    # Init traceback matrix
    paths = np.full([r + 1, c + 1], "", dtype='<U4')

    # Fill the scoring and traceback matrices
    last_under_max_dist = 0
    i1 = 0
    for i0 in range(r):
        i1 = i0 + 1
        if last_under_max_dist == -1:
            prev_last_under_max_dist = np.inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        for j0 in range(max(0, i0 - max(0, r - c) - window + 1), min(c, i0 + max(0, c - r) + window)):
            j1 = j0 + 1
            d, d_indel = fn(s1[i0], s2[j0])
            if max_step is not None:
                if d > max_step:
                    d = np.inf
                if d_indel > max_step:
                    d_indel = np.inf
                if d > max_step and d_indel > max_step:
                    continue
            # print(f"[{i1},{j1}] -> [{s1[i0]},{s2[j0]}] -> {d},{d_indel}")
            from_left_score = d_indel + scores[i1, j0] + penalty
            from_above_score = d_indel + scores[i0, j1] + penalty
            from_diag_score = d + scores[i0, j0]
            scores[i1, j1] = min(from_left_score, from_above_score, from_diag_score)
            # make note of which cell was best in the traceback array
            if scores[i1, j1] == from_left_score:
                paths[i1, j1] += Direction.LEFT.value
            if scores[i1, j1] == from_above_score:
                paths[i1, j1] += Direction.UP.value
            if scores[i1, j1] == from_diag_score:
                paths[i1, j1] += Direction.UP_LEFT.value
            if max_dist is not None:
                if scores[i1, j1] <= max_dist:
                    last_under_max_dist = j0
                else:
                    scores[i1, j1] = np.inf
                    if prev_last_under_max_dist < j1:
                        break
        if max_dist is not None and last_under_max_dist == -1:
            return np.inf, scores
    if psi == 0:
        d = scores[i1, min(c, c + window - 1)]
    else:
        ir, ic = i1, min(c, c + window - 1)
        vr, vc = scores[ir-psi:ir+1, ic], scores[ir, ic-psi:ic+1]
        mir, mic = np.argmin(vr), np.argmin(vc)
        if vr[mir] < vc[mic]:
            scores[ir-psi+mir+1:ir+1, ic] = -1
            d = vr[mir]
        else:
            scores[ir, ic-psi+mic+1:ic+1] = -1
            d = vc[mic]
    return d, scores, paths
