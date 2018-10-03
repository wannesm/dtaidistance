# -*- coding: UTF-8 -*-
"""
dtaidistance.dp
~~~~~~~~~~~~~~~

Generic Dynamic Programming functions

:author: Wannes Meert
:copyright: Copyright 2017-2018 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import numpy as np


logger = logging.getLogger("be.kuleuven.dtai.distance")


def dp(s1, s2, fn, border=None, window=None, max_dist=None,
       max_step=None, max_length_diff=None, penalty=None, psi=None):
    """
    Generic dynamic programming.

    This function does not optimize storage when a window size is given (e.g. in contrast with
    the fast DTW functions).

    :param s1: First sequence
    :param s2: Second sequence
    :param fn: Function to compare two items from both sequences
    :param border: Callable object to fill in the initial borders (border(row_idx, col_idx).
    :param window: see :meth:`distance`
    :param max_dist: see :meth:`distance`
    :param max_step: see :meth:`distance`
    :param max_length_diff: see :meth:`distance`
    :param penalty: see :meth:`distance`
    :param psi: see :meth:`distance`
    :returns: (DTW distance, DTW matrix)
    """
    r, c = len(s1), len(s2)
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
    dtw = np.full((r + 1, c + 1), np.inf)
    if border:
        for ci in range(c + 1):
            dtw[0, ci] = border(0, ci)
        for ri in range(1, r + 1):
            dtw[ri, 0] = border(ri, 0)
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
        for j in range(max(0, i - max(0, r - c) - window + 1), min(c, i + max(0, c - r) + window)):
            d, d_indel = fn(s1[i], s2[j])
            if max_step is not None:
                if d > max_step:
                    d = np.inf
                if d_indel > max_step:
                    d_indel = np.inf
                if d > max_step and d_indel > max_step:
                    continue
            # print(f"[{i1},{j+1}] -> [{s1[i]},{s2[j]}] -> {d},{d_indel}")
            dtw[i1, j + 1] = min(d + dtw[i0, j],
                                 d_indel + dtw[i0, j + 1] + penalty,
                                 d_indel + dtw[i1, j] + penalty)
            if max_dist is not None:
                if dtw[i1, j + 1] <= max_dist:
                    last_under_max_dist = j
                else:
                    dtw[i1, j + 1] = np.inf
                    if prev_last_under_max_dist < j + 1:
                        break
        if max_dist is not None and last_under_max_dist == -1:
            return np.inf, dtw
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
