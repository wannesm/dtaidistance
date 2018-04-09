# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw_c
~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW), C implementation.

:author: Wannes Meert
:copyright: Copyright 2017-2018 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import math
import numpy as np
cimport numpy as np
cimport cython
import cython
import ctypes
from cpython cimport array, bool
from cython import parallel
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free, abs, labs
from libc.stdio cimport printf
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, pow


logger = logging.getLogger("be.kuleuven.dtai.distance")


DTYPE = np.double
ctypedef np.double_t DTYPE_t
cdef double inf = np.inf

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def distance(np.ndarray[DTYPE_t, ndim=1] s1, np.ndarray[DTYPE_t, ndim=1] s2,
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0, int psi=0):
    """
    Dynamic Time Warping (keep compact matrix)
    :param s1: First sequence (np.array(np.float64))
    :param s2: Second sequence
    :param window: Only allow for shifts up to this amount away from the two diagonals
    :param max_dist: Stop if the returned values will be larger than this value
    :param max_step: Do not allow steps larger than this value
    :param max_length_diff: Max length difference between the two sequences
    :param penalty: Cost incurrend when performing compression or expansion

    Returns: DTW distance
    """
    assert s1.dtype == DTYPE and s2.dtype == DTYPE
    cdef int r = len(s1)
    cdef int c = len(s2)
    if max_length_diff != 0 and abs(r-c) > max_length_diff:
        return inf
    if window == 0:
        window = max(r, c)
    if max_step == 0:
        max_step = inf
    else:
        max_step *= max_step
    if max_dist == 0:
        max_dist = inf
    else:
        max_dist *= max_dist
    penalty *= penalty
    cdef int length = min(c + 1, abs(r - c) + 2 * (window - 1) + 1 + 1 + 1)
    cdef np.ndarray[DTYPE_t, ndim=2] dtw = np.full((2, length), inf)
    # dtw[0, 0] = 0
    cdef int i
    for i in range(psi + 1):
        dtw[0, i] = 0
    cdef double last_under_max_dist = 0
    cdef double prev_last_under_max_dist = inf
    cdef int skip = 0
    cdef int skipp = 0
    cdef int i0 = 1
    cdef int i1 = 0
    cdef DTYPE_t d
    for i in range(r):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        skipp = skip
        skip = max(0, i - window + 1)
        i0 = 1 - i0
        i1 = 1 - i1
        dtw[i1 ,:] = inf
        j_start = max(0, i - max(0, r - c) - window + 1)
        j_end = min(c, i + max(0, c - r) + window)
        if dtw.shape[1] == c+ 1:
            skip = 0
        if psi != 0 and j_start == 0 and i < psi:
            dtw[i1, 0] = 0
        for j in range(j_start, j_end):
            d = (s1[i] - s2[j])**2
            if d > max_step:
                continue
            dtw[i1, j + 1 - skip] = d + min(dtw[i0, j - skipp],
                                            dtw[i0, j + 1 - skipp] + penalty,
                                            dtw[i1, j - skip] + penalty)
            if dtw[i1, j + 1 - skip] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1, j + 1 - skip] = inf
                if prev_last_under_max_dist + 1 - skipp < j + 1 - skip:
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            return inf
        if psi != 0 and j_end == len(s2) and len(s1) - 1 - i <= psi:
            psi_shortest = min(psi_shortest, dtw[i1, length - 1])
    if psi == 0:
        d = math.sqrt(dtw[i1, min(c, c + window - 1) - skip])
    else:
        ic = min(c, c + window - 1) - skip
        vc = dtw[i1, ic - psi:ic + 1]
        d = min(np.min(vc), psi_shortest)
        d = math.sqrt(d)
    # print(dtw)
    return d


def distance_nogil(double[:] s1, double[:] s2,
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0, int psi=0):
    """DTW distance.

    See distance(). This calls a pure c dtw computation that avoids the GIL.
    :param s1: First sequence (buffer of doubles)
    :param s2: Second sequence (buffer of doubles)
    """
    #return distance_nogil_c(s1, s2, len(s1), len(s2),
    # If the arrays (memoryviews) are not C contiguous, the pointer will not point to the correct array
    if isinstance(s1, (np.ndarray, np.generic)):
        if not s1.base.flags.c_contiguous:
            logger.debug("Warning: Sequence 1 passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            s1 = s1.copy()
    if isinstance(s2, (np.ndarray, np.generic)):
        if not s2.base.flags.c_contiguous:
            logger.debug("Warning: Sequence 2 passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            s2 = s2.copy()
    return distance_nogil_c(&s1[0], &s2[0], len(s1), len(s2),
                            window, max_dist, max_step, max_length_diff, penalty, psi)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.infer_types(False)
cdef double distance_nogil_c(
             double *s1, double *s2,
             int r, # len_s1
             int c, # len_s2
             int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0, double penalty=0, int psi=0) nogil:
    """DTW distance.

    See distance(). This is a pure c dtw computation that avoid the GIL.
    """
    #printf("%i, %i\n", r, c)
    if max_length_diff != 0 and abs(r-c) > max_length_diff:
        return inf
    if window == 0:
        window = max(r, c)
    if max_step == 0:
        max_step = inf
    else:
        max_step = pow(max_step, 2)
    if max_dist == 0:
        max_dist = inf
    else:
        max_dist = pow(max_dist, 2)
    penalty = pow(penalty, 2)
    cdef int length = min(c+1,abs(r-c) + 2*(window-1) + 1 + 1 + 1)
    #printf("length (c) = %i\n", length)
    #cdef array.array dtw_tpl = array.array('d', [])
    #cdef array.array dtw
    #dtw = array.clone(dtw_tpl, length*2, zero=False)
    cdef double * dtw
    dtw = <double *> malloc(sizeof(double) * length * 2)
    cdef int i
    cdef int j
    for j in range(length*2):
        dtw[j] = inf
    # dtw[0] = 0
    for i in range(psi + 1):
        dtw[i] = 0
    cdef double last_under_max_dist = 0
    cdef double prev_last_under_max_dist = inf
    cdef int skip = 0
    cdef int skipp = 0
    cdef int i0 = 1
    cdef int i1 = 0
    cdef int minj
    cdef int maxj
    cdef double minv
    cdef DTYPE_t d
    cdef double tempv
    cdef double psi_shortest = inf
    cdef int iii
    for i in range(r):
        #
        #printf("[ ")
        #for iii in range(length):
        #    printf("%f ", dtw[iii])
        #printf("\n")
        #for iii in range(length,length*2):
        #    printf("%f ", dtw[iii])
        #printf("]\n")
        #
        if last_under_max_dist == -1:
            prev_last_under_max_dist = inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        maxj = r - c
        if maxj < 0:
            maxj = 0
        maxj = i - maxj - window + 1
        if maxj < 0:
            maxj = 0
        skipp = skip
        skip = maxj
        i0 = 1 - i0
        i1 = 1 - i1
        for j in range(length):
            dtw[length * i1 + j] = inf
        if length == c + 1:
            skip = 0
        minj = c - r
        if minj < 0:
            minj = 0
        minj = i + minj + window
        if minj > c:
            minj = c
        if psi != 0 and maxj == 0 and i < psi:
            dtw[i1*length + 0] = 0
        for j in range(maxj, minj):
            #printf('s1[i] = s1[%i] = %f , s2[j] = s2[%i] = %f\n', i, s1[i], j, s2[j])
            d = pow(s1[i] - s2[j], 2)
            if d > max_step:
                continue
            minv = dtw[i0*length + j - skipp]
            tempv = dtw[i0*length + j + 1 - skipp] + penalty
            if tempv < minv:
                minv = tempv
            tempv = dtw[i1*length + j - skip] + penalty
            if tempv < minv:
                minv = tempv
            #printf('d = %f, minv = %f\n', d, minv)
            dtw[i1 * length + j + 1 - skip] = d + minv
            #
            #printf('%i, %i, %i\n',i0*length + j - skipp,i0*length + j + 1 - skipp,i1*length + j - skip)
            #printf('%f, %f, %f\n',dtw[i0*length + j - skipp],dtw[i0*length + j + 1 - skipp],dtw[i1*length + j - skip])
            #printf('i=%i, j=%i, d=%f, skip=%i, skipp=%i\n',i,j,d,skip,skipp)
            #printf("[ ")
            #for iii in range(length):
            #    printf("%f ", dtw[iii])
            #printf("\n")
            #for iii in range(length,length*2):
            #    printf("%f ", dtw[iii])
            #printf("]\n")
            #
            if dtw[i1*length + j + 1 - skip] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1*length + j + 1 - skip] = inf
                if prev_last_under_max_dist + 1 - skipp < j + 1 - skip:
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            return inf

        if psi != 0 and minj == c and r - 1 - i <= psi:
            if dtw[i1*length + length - 1] < psi_shortest:
                psi_shortest = dtw[i1*length + length - 1]

        # printf("[ ")
        # for iii in range(i1*length,i1*length + length):
        #    printf("%f ", dtw[iii])
        # printf("]\n")

    # print(dtw)
    if window - 1 < 0:
        c = c + window - 1
    cdef double result = sqrt(dtw[length * i1 + c - skip])
    if psi != 0:
        for i in range(c - skip - psi, c - skip + 1):  # iterate over vci
            if dtw[i1*length + i] < psi_shortest:
                psi_shortest = dtw[i1*length + i]
        result = sqrt(psi_shortest)
    free(dtw)
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def distance_matrix(cur, double max_dist=inf, int max_length_diff=0,
                    int window=0, double max_step=0, double penalty=0, block=None, **kwargs):
    """Compute a distance matrix between all sequences given in `cur`.
    """
    if max_length_diff == 0:
        max_length_diff = 999999
    if block is None:
        block = ((0, len(cur)), (0, len(cur)))
    cdef double large_value = inf
    cdef np.ndarray[DTYPE_t, ndim=2] dists = np.zeros((len(cur), len(cur))) + large_value
    for r in range(block[0][0], block[0][1]):
        for c in range(max(r + 1, block[1][0]), block[1][1]):
            if labs(len(cur[r]) - len(cur[c])) <= max_length_diff:
                dists[r, c] = distance(cur[r], cur[c], window=window,
                                       max_dist=max_dist, max_step=max_step,
                                       max_length_diff=max_length_diff,
                                       penalty=penalty)
    return dists


def distance_matrix_nogil(cur, double max_dist=inf, int max_length_diff=0,
                          int window=0, double max_step=0, double penalty=0, block=None,
                          bool is_parallel=False, **kwargs):
    """Compute a distance matrix between all sequences given in `cur`.
    This method calls a pure c implementation of the dtw computation that
    avoids the GIL.
    """
    # https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
    # Prepare for only c datastructures
    cdef int block_rb=0
    cdef int block_re=0
    cdef int block_cb=0
    cdef int block_ce=0
    if block is not None:
        block_rb = block[0][0]
        block_re = block[0][1]
        block_cb = block[1][0]
        block_ce = block[1][1]
    if max_length_diff == 0:
        max_length_diff = 999999
    cdef double large_value = inf
    dists_py = np.zeros((len(cur), len(cur))) + large_value
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] dists = dists_py
    #print('dists: {}, {}'.format(dists_py.shape, dists_py.shape[0]*dists_py.shape[1]))
    cdef double **cur2 = <double **> malloc(len(cur) * sizeof(double*))
    cdef int *cur2_len = <int *> malloc(len(cur) * sizeof(int))
    cdef long ptr;
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] cur_np;
    #for i in range(len(cur)):
    #    print(cur[i])
    if cur.__class__.__name__ == "SeriesContainer":
        cur = cur.c_data()
    if type(cur) in [list, set]:
        for i in range(len(cur)):
            ptr = cur[i].ctypes.data
            cur2[i] = <double *> ptr
            cur2_len[i] = len(cur[i])
    elif isinstance(cur, np.ndarray):
        if not cur.flags.c_contiguous:
            logger.debug("Warning: The numpy array or matrix passed to method distance_matrix is not C-contiguous. " +
                         "The array will be copied.")
            cur = cur.copy(order='C')
        cur_np = cur
        for i in range(len(cur)):
            cur2[i] = &cur_np[i,0]
            cur2_len[i] = cur_np.shape[1]
    else:
        return None
    if is_parallel:
        distance_matrix_nogil_c_p(cur2, len(cur), cur2_len, &dists[0,0], max_dist, max_length_diff, window, max_step, penalty,
                                  block_rb, block_re, block_cb, block_ce)
    else:
        distance_matrix_nogil_c(cur2, len(cur), cur2_len, &dists[0,0], max_dist, max_length_diff, window, max_step, penalty,
                                block_rb, block_re, block_cb, block_ce)
    free(cur2)
    free(cur2_len)
    return dists_py


def distance_matrix_nogil_p(cur, double max_dist=inf, int max_length_diff=0,
                            int window=0, double max_step=0, double penalty=0, block=None, **kwargs):
    """Compute a distance matrix between all sequences given in `cur`.
    This method calls a pure c implementation of the dtw computation that
    avoids the GIL and executes them in parallel.
    """
    return distance_matrix_nogil(cur, max_dist=max_dist, max_length_diff=max_length_diff,
                                 window=window, max_step=max_step, penalty=penalty, block=block,
                                 is_parallel=True, **kwargs)


cdef distance_matrix_nogil_c(double **cur, int len_cur, int* cur_len, double* output,
                             double max_dist=0, int max_length_diff=0,
                             int window=0, double max_step=0, double penalty=0,
                             int block_rb=0, int block_re=0, int block_cb=0, int block_ce=0):
    #for i in range(len_cur):
    #    print(i)
    #    print(cur_len[i])
    #    for j in range(cur_len[i]):
    #        printf("%f ", cur[i][j])
    #    printf("\n")
    #printf("---\n")
    cdef int r
    cdef int c
    cdef int cb

    if block_re == 0:
        block_re = len_cur
    if block_ce == 0:
        block_ce = len_cur
    for r in range(block_rb, block_re):
        if r + 1 > block_cb:
            cb = r+1
        else:
            cb = block_cb
        for c in range(cb, block_ce):
            output[len_cur*r + c] = distance_nogil_c(cur[r], cur[c], cur_len[r], cur_len[c],
                                                     window=window, max_dist=max_dist,
                                                     max_step=max_step, max_length_diff=max_length_diff,
                                                     penalty=penalty)
            #for i in range(len_cur):
            #    for j in range(len_cur):
            #        printf("%f ", output[i*len_cur+j])
            #    printf("\n")
            #printf("---\n")


cdef distance_matrix_nogil_c_p(double **cur, int len_cur, int* cur_len, double* output,
                               double max_dist=0, int max_length_diff=0,
                               int window=0, double max_step=0, double penalty=0,
                               int block_rb=0, int block_re=0, int block_cb=0, int block_ce=0):
    # Requires openmp which is not supported for clang on mac by default (use newer version of clang)
    cdef Py_ssize_t r
    cdef Py_ssize_t c
    cdef Py_ssize_t cb
    cdef Py_ssize_t brb = block_rb  # TODO: why is this necessary for cython?

    # printf("%f %f %f %f", block_rb, block_re, block_cb, block_ce)

    if block_re == 0:
        block_re = len_cur
    if block_ce == 0:
        block_ce = len_cur
    with nogil, parallel():
        for r in prange(brb, block_re):
            if r + 1 > block_cb:
                cb = r+1
            else:
                cb = block_cb
            for c in range(cb, block_ce):
                output[len_cur*r + c] = distance_nogil_c(cur[r], cur[c], cur_len[r], cur_len[c],
                                                         window=window, max_dist=max_dist,
                                                         max_step=max_step, max_length_diff=max_length_diff,
                                                         penalty=penalty)
