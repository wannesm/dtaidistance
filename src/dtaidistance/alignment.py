# -*- coding: UTF-8 -*-
"""
dtaidistance.alignment
~~~~~~~~~~~~~~~~~~~~~~

Sequence alignment (e.g. Needleman–Wunsch).

:author: Wannes Meert
:copyright: Copyright 2017-2022 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
from .dp import dp, Direction

try:
    import numpy as np
except ImportError:
    np = None


def needleman_wunsch(s1, s2, window=None, max_dist=None,
                     max_step=None, max_length_diff=None, psi=None,
                     substitution=None):
    """Needleman-Wunsch global sequence alignment.

    Needleman-Wunsch finds the optimal aligment between two sequences by maximizing similarity.

    For background on this algorithm, see the original paper:
    Needleman, Saul B. & Wunsch, Christian D. (1970).
    "A general method applicable to the search for similarities in the amino acid sequence of two proteins".
    Journal of Molecular Biology. 48 (3): 443–53. doi:10.1016/0022-2836(70)90057-4 .

    It is equivalent to the Levenshtein distance that tries to minimize the edit distance. For more information see:
    Sellers PH (1974). "On the theory and computation of evolutionary distances".
    SIAM Journal on Applied Mathematics. 26 (4): 787–793. doi:10.1137/0126070 .

    :param s1: First sequence
    :param s2: Second sequence
    :param window:
    :param max_dist: Stop warping path if distance exceeds this value
    :param max_step: Stop warping path if the last increase exceeds this value.
    :param max_length_diff: Exit if the two sequences differ in length.
    :param psi: Psi relaxation
    :param substitution: Function with signature (s1[i], s2[j]) that returns the value to add to the
        current warping path given two symbols. For Needleman-Wunsch the default function that is used
         is -1 for a match and +1 for a mismatch.
         If you have custom distance values between particular symbols, you can use the `make_substitution_fn`
         method to generate a function from a dictionary.

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
        substitution = _default_substitution_fn
    value, scores, paths = dp(s1, s2,
                       fn=substitution, border=_needleman_wunsch_border,
                       penalty=0, window=window, max_dist=max_dist,
                       max_step=max_step, max_length_diff=max_length_diff, psi=psi)
    return -value, -scores, paths


def _needleman_wunsch_border(ri, ci):
    if ri == 0:
        return ci
    if ci == 0:
        return ri
    return 0


def _default_substitution_fn(v1, v2):
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
        For example `matrix={('A','B'): 2, ('B', 'A'): 3}`.
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
        The default order is 0,1,2: diagonal, up, left
        For example, 1,0,2 is up, diagonal, left
        There might be more optimal paths than covered by these orderings. For example,
        when using combinations of these orderings in different parts of the matrix.
    """
    i, j = int(paths.shape[0] - 1), int(paths.shape[1] - 1)
    p = [(i - 1, j - 1)]
    ops = [(-1,-1), (-1,-0), (-0,-1)]
    op_chars = [Direction.UP_LEFT.value, Direction.UP.value, Direction.LEFT.value]
    if order is None:
        order = [0, 1, 2]
    while i > 0 and j > 0:
        opi, opj = next(
            ops[orderi]
            for orderi in order
            if op_chars[orderi] in paths[i, j]
        )
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
