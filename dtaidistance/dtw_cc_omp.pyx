"""
dtaidistance.dtw_cc_omp
~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW), C implementation, with OpenMP support.

:author: Wannes Meert
:copyright: Copyright 2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
from cpython cimport array
import array
cimport dtw_cc
cimport dtaidistancec
cimport dtaidistancec_omp


def distance_matrix(cur, block=None, **kwargs):
    """Compute a distance matrix between all sequences given in `cur`.
    This method calls a pure c implementation of the dtw computation that
    avoids the GIL.

    Assumes C-contiguous arrays.

    :param cur: DTWSeriesMatrix or DTWSeriesPointers
    :param block: see DTWBlock
    :param kwargs: Settings (see DTWSettings)
    :return: The distance matrix as a list representing the triangular matrix.
    """
    cdef int length = 0
    cdef int block_rb=0
    cdef int block_re=0
    cdef int block_cb=0
    cdef int block_ce=0
    cdef ri = 0
    if block is not None and block != 0.0:
        block_rb = block[0][0]
        block_re = block[0][1]
        block_cb = block[1][0]
        block_ce = block[1][1]

    settings = dtw_cc.DTWSettings(**kwargs)
    block = dtw_cc.DTWBlock(rb=block_rb, re=block_re, cb=block_cb, ce=block_ce)
    length = dtaidistancec.dtw_distances_length(block._block, len(cur))

    cdef array.array dists = array.array('d')
    dists.resize(length)

    if isinstance(cur, dtw_cc.DTWSeriesMatrix) or isinstance(cur, dtw_cc.DTWSeriesPointers):
        pass
    elif cur.__class__.__name__ == "SeriesContainer":
        cur = cur.c_data()
    else:
        cur = dtw_cc.dtw_series_from_data(cur)

    if isinstance(cur, dtw_cc.DTWSeriesPointers):
        dtaidistancec_omp.dtw_distances_ptrs_parallel(
            cur.ptrs, cur.nb_ptrs, cur.lengths, dists.as_doubles, &block._block, &settings._settings)
    elif isinstance(cur, dtw_cc.DTWSeriesMatrix):
        dtaidistancec_omp.dtw_distances_matrix_parallel(
            cur.matrix, cur.nb_rows, cur.nb_cols, dists.as_doubles, &block._block, &settings._settings)

    return dists
