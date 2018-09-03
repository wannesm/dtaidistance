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

from .dp import dp


def needleman_wunsch(s1, s2, window=None, max_dist=None,
                     max_step=None, max_length_diff=None, psi=None):
    """Needleman-Wunsch global sequence alignment.

    Example:

        >> s1 = "GATTACA"
        >> s2 = "GCATGCU"
        >> value, matrix = alignment.needleman_wunsch(s1, s2)
        >> algn, s1a, s2a = alignment.best_alignment(matrix, s1, s2)
        >> print(matrix)
           [[-0., -1., -2., -3., -4., -5., -6., -7.],
            [-1.,  1., -0., -1., -2., -3., -4., -5.],
            [-2., -0., -0.,  1., -0., -1., -2., -3.],
            [-3., -1., -1., -0.,  2.,  1., -0., -1.],
            [-4., -2., -2., -1.,  1.,  1., -0., -1.],
            [-5., -3., -3., -1., -0., -0., -0., -1.],
            [-6., -4., -2., -2., -1., -1.,  1., -0.],
            [-7., -5., -3., -1., -2., -2., -0., -0.]]
        >> print(''.join(s1a), ''.join(s2a))
            'G-ATTACA', 'GCAT-GCU'

    """
    value, matrix = dp(s1, s2,
                       _needleman_wunsch_fn, border=_needleman_wunsch_border,
                       penalty=0, window=window, max_dist=max_dist,
                       max_step=max_step, max_length_diff=max_length_diff, psi=psi)
    matrix = -matrix
    return value, matrix


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


def best_alignment(paths, s1=None, s2=None, gap="-", order=None):
    """Compute the optimal alignment from the nxm paths matrix.

    :param paths: Paths matrix (e.g. from needleman_wunsch)
    :param s1: First sequence, if given the aligned sequence will be created
    :param s2: Second sequence, if given the aligned sequence will be created
    :param gap: Gap symbol that is inserted into s1 and s2 to align the sequences
    :param order: Array with order of comparisons (there might be multiple optimal paths)
        The default order is 0,1,2: (-1,-1), (-1,-0), (-0,-1)
        For example, 1,0,2 is (-1,-0), (-1,-1), (-0,-1)
        There might be more optimal paths than covered by these orderings. For example,
        when using combinations of these orderings in different parts of the matrix.
    """
    i, j = int(paths.shape[0] - 1), int(paths.shape[1] - 1)
    p = []
    if paths[i, j] != -1:
        p.append((i - 1, j - 1))
    ops = [(-1,-1), (-1,-0), (-0,-1)]
    if order is None:
        order = [0, 1, 2]
    while i > 0 and j > 0:
        prev_vals = [paths[i + ops[orderi][0], j + ops[orderi][1]] for orderi in order]
        # c = np.argmax([paths[i - 1, j - 1], paths[i - 1, j], paths[i, j - 1]])
        c = int(np.argmax(prev_vals))
        opi, opj = ops[order[c]]
        i, j = i + opi, j + opj
        if paths[i, j] != -1:
            p.append((i - 1, j - 1))
    p.pop()
    p.reverse()
    if s1 is not None:
        s1a = []
        s1ip = -1
        for s1i, _ in p:
            if s1i == s1ip + 1:
                s1a.append(s1[s1i])
            else:
                s1a.append(gap)
            s1ip = s1i
    else:
        s1a = None
    if s2 is not None:
        s2a = []
        s2ip = -1
        for _, s2i in p:
            if s2i == s2ip + 1:
                s2a.append(s2[s2i])
            else:
                s2a.append(gap)
            s2ip = s2i
    else:
        s2a = None

    return p, s1a, s2a
