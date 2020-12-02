from cython import Py_ssize_t
from cython.parallel import parallel, prange
cimport numpy as np
import numpy as np

from .dtw_cc cimport DTWSeriesMatrix, DTWSettings

cdef extern from "Python.h":
    Py_ssize_t PY_SSIZE_T_MAX

cimport dtaidistancec_dtw_search
cimport dtaidistancec_dtw


def lb_keogh_envelope(double[:] s1,  double[:] L, double[:] U, **kwargs):
    settings = DTWSettings(**kwargs)
    dtaidistancec_dtw_search.lower_upper_ascending_minima(&s1[0], len(s1), &L[0], &U[0], &settings._settings)


def lb_keogh_envelope_parallel(s1, **kwargs):
    settings = DTWSettings(**kwargs)
    cdef double[:,:] L = np.empty((len(s1), len(s1[0])), dtype=np.double)
    cdef double[:,:] U = np.empty((len(s1), len(s1[0])), dtype=np.double)
    cdef DTWSeriesMatrix matrix = s1.c_data_compat()
    cdef int d
    cdef int l = len(s1)
    for d in prange(l, nogil=True):
        dtaidistancec_dtw_search.lower_upper_ascending_minima(&matrix._data[d,0], matrix._data.shape[1], &L[d,0], &U[d,0], &settings._settings)
    return list(zip(list(L.base), list(U.base)))


def lb_keogh_from_envelope(double[:] s1, s2_envelope, **kwargs):
    settings = DTWSettings(**kwargs)
    cdef double[:] L = s2_envelope[0]
    cdef double[:] U = s2_envelope[1]
    cdef double lb = dtaidistancec_dtw_search.lb_keogh_from_envelope(&s1[0], len(s1), &L[0], &U[0], &settings._settings)
    return lb


def lb_keogh_from_envelope_parallel(s1,  s2_envelope, **kwargs):
    settings = DTWSettings(**kwargs)
    cdef int l1 = len(s1)
    cdef int l2 = len(s2_envelope)
    cdef int i, j
    cdef double[:,:] lb = np.zeros((l1, l2))
    cdef DTWSeriesMatrix matrix1 = s1.c_data_compat()
    cdef double[:,:] L = np.vstack([l for (l,_) in s2_envelope])
    cdef double[:,:] U = np.vstack([u for (_,u) in s2_envelope])
    with nogil:
        for i in prange(l1):
            for j in range(l2):
                lb[i,j] = dtaidistancec_dtw_search.lb_keogh_from_envelope(&matrix1._data[i,0], matrix1._data.shape[1], &L[j,0], &U[j,0], &settings._settings)
    return lb.base


def lb_keogh(double[:] s1,  double[:] s2, **kwargs):
    settings = DTWSettings(**kwargs)
    cdef double lb = dtaidistancec_dtw.lb_keogh(&s1[0], len(s1), &s2[0], len(s2), &settings._settings)
    return lb


def lb_keogh_parallel(s1, s2, **kwargs):
    settings = DTWSettings(**kwargs)
    cdef int l1 = len(s1)
    cdef int l2 = len(s2)
    cdef int i, j
    cdef double[:,:] lb = np.zeros((l1, l2))
    cdef DTWSeriesMatrix matrix1 = s1.c_data_compat()
    cdef DTWSeriesMatrix matrix2 = s2.c_data_compat()
    with nogil:
        for i in prange(l1):
            for j in range(l2):
                lb[i,j] = dtaidistancec_dtw.lb_keogh(&matrix1._data[i,0], matrix1._data.shape[1], &matrix2._data[j,0], matrix2._data.shape[1], &settings._settings)
    return lb.base


def nearest_neighbour_lb_keogh(data, double[:] query, double[:] lb, **kwargs):
    cdef int verbose = 0
    cdef Py_ssize_t location = 0
    cdef double distance = 0
    cdef DTWSeriesMatrix matrix = data.c_data_compat()
    settings = DTWSettings(**kwargs)
    dtaidistancec_dtw_search.nn_lb_keogh(&matrix._data[0,0], matrix.nb_rows*matrix.nb_cols, &query[0], &lb[0], len(query), verbose, &location, &distance, &settings._settings)
    return location, distance


def nearest_neighbour_lb_keogh_parallel(data, queries, double[:,:] lb, **kwargs):
    cdef int verbose = 0
    settings = DTWSettings(**kwargs)
    cdef int nq = len(queries)
    cdef int lq = len(queries[0])
    cdef int i
    cdef Py_ssize_t[:] locations = np.zeros(nq, dtype=np.intp)
    cdef double[:] distances = np.ones(nq) * np.inf
    cdef DTWSeriesMatrix matrix = data.c_data_compat()
    cdef DTWSeriesMatrix qmatrix = queries.c_data_compat()
    for i in prange(nq, nogil=True):
        dtaidistancec_dtw_search.nn_lb_keogh(&matrix._data[0,0], matrix._data.shape[0]*matrix._data.shape[1], &qmatrix._data[i,0], &lb[0,i], lq, verbose, &locations[i], &distances[i], &settings._settings)
    return locations.base


def nearest_neighbour_ucr(data, double[:] query, **kwargs):
    cdef int verbose = 0
    cdef int skip = 1
    cdef Py_ssize_t location = 0
    cdef double distance = 0
    settings = DTWSettings(**kwargs)
    cdef DTWSeriesMatrix matrix = data.c_data_compat()
    dtaidistancec_dtw_search.ucrdtw(&matrix._data[0,0], matrix.nb_rows*matrix.nb_cols, skip, &query[0], len(query), verbose, &location, &distance, &settings._settings)
    return location, distance


def nearest_neighbour_lb_keogh_subsequence(double[:] data, double[:] query, double[:] l, double[:] u, bint use_ucr = True, **kwargs):
    cdef bint verbose = False
    cdef Py_ssize_t location = 0
    cdef double distance = 0
    settings = DTWSettings(**kwargs)
    if use_ucr:
        dtaidistancec_dtw_search.ucrdtw(&data[0], len(data), False, &query[0], len(query), verbose, &location, &distance, &settings._settings)
    else:
        dtaidistancec_dtw_search.nn_lb_keogh_subsequence(&data[0], len(data), &query[0], &l[0], &u[0], len(query), verbose, &location, &distance, &settings._settings)
    return location, distance


def nearest_neighbour_lb_keogh_subsequence_parallel(double[:] data, queries, envelopes, bint use_ucr = True, **kwargs):
    cdef int verbose = 0
    settings = DTWSettings(**kwargs)
    cdef int nq = len(queries)
    cdef int lq = len(queries[0])
    cdef int i
    cdef Py_ssize_t[:] locations = np.zeros(nq, dtype=int)
    cdef double[:] distances = np.ones(nq) * np.inf
    cdef DTWSeriesMatrix qmatrix = queries.c_data_compat()
    cdef double[:,:] L = np.vstack([l for (l,_) in envelopes])
    cdef double[:,:] U = np.vstack([u for (_,u) in envelopes])
    for i in prange(nq, nogil=True):
        if use_ucr:
            dtaidistancec_dtw_search.ucrdtw(&data[0], data.shape[0], False, &qmatrix._data[i,0], lq, verbose, &locations[i], &distances[i], &settings._settings)
        else:
            dtaidistancec_dtw_search.nn_lb_keogh_subsequence(&data[0], data.shape[0], &qmatrix._data[i,0], &L[i,0], &U[i,0], lq, verbose, &locations[i], &distances[i], &settings._settings)
    return locations.base
