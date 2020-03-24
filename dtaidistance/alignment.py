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
                     max_step=None, max_length_diff=None, psi=None,
                     substitution=None):
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
    if substitution is None:
        substitution =  _default_substitution_fn
    value, matrix = dp(s1, s2,
                       fn=substitution, border=_needleman_wunsch_border,
                       penalty=0, window=window, max_dist=max_dist,
                       max_step=max_step, max_length_diff=max_length_diff, psi=psi)
    matrix = -matrix
    return value, matrix



def _needleman_wunsch_border(ri, ci):
    if ri == 0:
        return ci
    if ci == 0:
        return ri
    return 0


def  _default_substitution_fn(v1, v2):
    """Default substitution function.

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


def make_substitution_fn(matrix, gap=1, opt='max'):
    """Make a similarity function from a dictionary.
    
    Elements that are not in the dictionary are passed to the default
    function. This allows for this function to be used for only
    using the gap penalty as follows.

        substitution = make_substitution_fn({}, gap=0.5)

    :param matrix: Substitution matrix as a dictionary of tuples to values.
    :param opt: Direction in which matrix optimises alignments. If `max`,
        values are reversed, see :meth:` _default_substitution_fn`.
    :return: Function that compares two elements.
    """

    if opt == 'max':
        modifier = -1.0
    else:
        modifier = 1.0

    def _unwrap(a, b):
        if (a, b) in matrix:
            return matrix[(a, b)] * modifier, gap
        elif (b, a) in matrix:
            return matrix[(b, a)] * modifier, gap
        else:
            return _default_substitution_fn(a, b)[0], gap

    return _unwrap


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
    p = [(i - 1, j - 1)]
    ops = [(-1,-1), (-1,-0), (-0,-1)]
    if order is None:
        order = [0, 1, 2]
    while i > 0 and j > 0:
        prev_vals = [paths[i + ops[orderi][0], j + ops[orderi][1]] for orderi in order]
        # c = np.argmax([paths[i - 1, j - 1], paths[i - 1, j], paths[i, j - 1]])
        c = int(np.argmax(prev_vals))
        # print(f"{i},{j}: {prev_vals} -> {c} ({ops[order[c]]})")
        opi, opj = ops[order[c]]
        i, j = i + opi, j + opj
        p.append((i - 1, j - 1))
    while i > 0:
        i -= 1
        p.append((i -1, j - 1))
    while j > 0:
        j -= 1
        p.append((i -1, j - 1))

    s1a = None if s1 is None else []
    s2a = None if s2 is None else []
    s1ip, s2ip = p[0]
    for s1i, s2i in p[1:]:
        if s1i != s1ip and s2i != s2ip:
            # diagonal
            if s1a is not None:
                s1a.append(s1[s1ip])
            if s2a is not None:
                s2a.append(s2[s2ip])
        elif s1i == s1ip:
            if s1a is not None:
                s1a.append(gap)
            if s2a is not None:
                s2a.append(s2[s2ip])
        elif s2i == s2ip:
            if s1a is not None:
                s1a.append(s1[s1ip])
            if s2a is not None:
                s2a.append(gap)
        s1ip, s2ip = s1i, s2i
    if s1a is not None:
        s1a.reverse()
    if s2a is not None:
        s2a.reverse()

    p.pop()
    p.reverse()
    return p, s1a, s2a
