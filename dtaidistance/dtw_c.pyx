import numpy as np
cimport numpy as np
cimport cython
from cython import parallel
from cython.parallel import prange

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
cdef double inf = np.inf

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def distance(np.ndarray[DTYPE_t, ndim=1] s1, np.ndarray[DTYPE_t, ndim=1] s2,
             int normalized=0, int window=0, double max_dist=0,
             double max_step=0, int max_length_diff=0):
    """
    Dynamic Time Warping (keep compact matrix)
    :param s1: First sequence
    :param s2: Second sequence
    :param dist_func: Point-wise distance
    :param normalized: (Not used)
    :param window: Only allow for shifts up to this amount away from the two diagonals
    :param max_dist: Stop if the returned values will be larger than this value
    :param max_step: Do not allow steps larger than this value
    :param compact: Use compact storage (2 by abs(r-c)*2*(window-1)+3)

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
    if max_dist == 0:
        max_dist = inf
    cdef np.ndarray[DTYPE_t, ndim=2] dtw = np.full((2, min( c +1 ,abs( r -c ) + 2 *( window -1 ) + 1 + 1 +1)), inf)
    dtw[0, 0] = 0
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
        if dtw.shape[1] == c+ 1:
            skip = 0
        for j in range(max(0, i - max(0, r - c) - window + 1), min(c, i + max(0, c - r) + window)):
            d = abs(s1[i] - s2[j])
            if d > max_step:
                continue
            dtw[i1, j + 1 - skip] = d + min(dtw[i0, j - skipp], dtw[i0, j + 1 - skipp], dtw[i1, j - skip])
            if dtw[i1, j + 1 - skip] <= max_dist:
                last_under_max_dist = j
            else:
                dtw[i1, j + 1 - skip] = inf
                if prev_last_under_max_dist < j + 1:
                    break
        if last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            return inf
    # print(dtw)
    return dtw[i1, min(c, c + window - 1) - skip]


#cdef double distance_p(double[:,:] dtw,
#             double[:] s1, double[:] s2,
#             int len_s1, int len_s2,
#             int normalized=0, int window=0, double max_dist=0,
#             double max_step=0, int max_length_diff=0) nogil:
#    cdef int r = len_s1
#    cdef int c = len_s2
#    if max_length_diff != 0 and abs(r-c) > max_length_diff:
#        return inf
#    if window == 0:
#        window = max(r, c)
#    if max_step == 0:
#        max_step = inf
#    if max_dist == 0:
#        max_dist = inf
#    dtw[0, 0] = 0
#    cdef double last_under_max_dist = 0
#    cdef double prev_last_under_max_dist = inf
#    cdef int skip = 0
#    cdef int skipp = 0
#    cdef int i0 = 1
#    cdef int i1 = 0
#    cdef int i
#    cdef int j
#    cdef DTYPE_t d
#    for i in range(r):
#        if last_under_max_dist == -1:
#            prev_last_under_max_dist = inf
#        else:
#            prev_last_under_max_dist = last_under_max_dist
#        last_under_max_dist = -1
#        skipp = skip
#        skip = max(0, i - window + 1)
#        i0 = 1 - i0
#        i1 = 1 - i1
#        dtw[i1 ,:] = inf
#        if dtw.shape[1] == c+ 1:
#            skip = 0
#        for j in range(max(0, i - max(0, r - c) - window + 1), min(c, i + max(0, c - r) + window)):
#            d = abs(s1[i] - s2[j])
#            if d > max_step:
#                continue
#            dtw[i1, j + 1 - skip] = d + min(dtw[i0, j - skipp], dtw[i0, j + 1 - skipp], dtw[i1, j - skip])
#            if dtw[i1, j + 1 - skip] <= max_dist:
#                last_under_max_dist = j
#            else:
#                dtw[i1, j + 1 - skip] = inf
#                if prev_last_under_max_dist < j + 1:
#                    break
#        if last_under_max_dist == -1:
#            # print('early stop')
#            # print(dtw)
#            return inf
#    # print(dtw)
#    return dtw[i1, min(c, c + window - 1) - skip]



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def distance_matrix(cur, double max_dist=inf, int max_diff_length=5,
                    int window=0, double max_point_dist=0, **kwargs):
    """Merge sequences.
    """
    if max_diff_length == 0:
        max_diff_length = 999999
    cdef double large_value = inf
    cdef np.ndarray[DTYPE_t, ndim=2] dists = np.zeros((len(cur), len(cur))) + large_value
    for r in range(len(cur)):
        for c in range(r + 1, len(cur)):
            if abs(len(cur[r]) - len(cur[c])) <= max_diff_length:
                dists[r, c] = distance(cur[r], cur[c], window=window,
                                       max_dist=max_dist, max_step=max_point_dist,
                                       max_length_diff=max_diff_length)
    return dists


#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function
#def distances_p(cur, double max_dist=inf, int max_diff_length=5,
#              int window=0, double max_point_dist=0, int num_threads=4):
#    """Merge sequences.
#    """
#    if max_diff_length == 0:
#        max_diff_length = 999999
#    cdef double large_value = inf
#    cdef np.ndarray[DTYPE_t, ndim=2] dists = np.zeros((len(cur), len(cur))) + large_value
#    # cdef double [:] s2_view
#    cdef int start
#    cdef int end = len(cur)
#    cdef int ri
#    cdef int ci
#    cdef double [:,:] dists_view = dists
#    cdef double[:] s1
#    cdef double[:] s2
#    cdef int len_s1
#    cdef int len_s2
#    cdef np.ndarray[DTYPE_t, ndim = 2] dtw
#    for ri in prange(end, nogil=True, num_threads=num_threads):
#        start = ri + 1
#        for ci in range(start, end):
#            # with nogil, parallel.parallel(num_threads=4):
#            with gil:
#                #s1 = cur[ci][1]
#                #s2 = cur[ri][1]
#                #len_s1 = len(cur[ci][1])
#                #len_s2 = len(cur[ri][1])
#                dtw = np.full((2, min(getl(cur,ri) + 1, abs(getl(cur,ci) - getl(cur,ri)) + 2 * (window - 1) + 1 + 1 + 1)), inf)
#            # if abs(len(cur[r][1]) - len(cur[c][1])) <= max_diff_length:
#            dists_view[ri, ci] = distance_p(dtw, gets(cur,ci), gets(cur,ri), getl(cur,ci), getl(cur,ri),
#                                   window=window,
#                                   max_dist=max_dist, max_step=max_point_dist,
#                                   max_length_diff=max_diff_length)
#    return dists
#
#cdef gets(cur, i) nogil:
#    with gil:
#        return cur[i][1]
#
#cdef getl(cur, i) nogil:
#    with gil:
#        return len(cur[i][1])