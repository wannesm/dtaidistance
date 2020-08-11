"""
dtaidistance.ed_cc
~~~~~~~~~~~~~~~~~~

Euclidean Distance (ED), C implementation.

:author: Wannes Meert
:copyright: Copyright 2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
cimport dtaidistancec_ed


logger = logging.getLogger("be.kuleuven.dtai.distance")


def distance(double[:] s1, double[:] s2):
    """ Euclidean distance between two sequences. Supports different lengths.

    If the two series differ in length, compare the last element of the shortest series
    to the remaining elements in the longer series.

    :param s1: Sequence of numbers
    :param s2: Sequence of numbers
    :return: Euclidean distance
    """
    return dtaidistancec_ed.euclidean_distance(&s1[0], len(s1), &s2[0], len(s2))


def distance_ndim(double[:, :] s1, double[:, :] s2):
    """ Euclidean distance between two sequences. Supports different lengths.

    If the two series differ in length, compare the last element of the shortest series
    to the remaining elements in the longer series.

    :param s1: Sequence of numbers
    :param s2: Sequence of numbers
    :return: Euclidean distance
    """
    # Assumes C contiguous
    if s1.shape[1] != s2.shape[1]:
        raise Exception(f"Dimension of sequence entries needs to be the same: {s1.shape[1]} != {s2.shape[1]}")
    ndim = s1.shape[1]
    return dtaidistancec_ed.euclidean_distance_ndim(&s1[0,0], len(s1), &s2[0,0], len(s2), ndim)
