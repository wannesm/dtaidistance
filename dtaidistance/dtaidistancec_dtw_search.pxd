
from dtaidistancec_dtw cimport DTWSettings

cdef extern from "dd_dtw_search.h":

    void lower_upper_lemire(double *t, int len, double *l, double *u, DTWSettings *settings) nogil
    double lb_keogh_from_envelope(double *s1, int l1, double *l, double *u, DTWSettings *settings) nogil
    int ucrdtw(double* data, long long data_size, int skip, double* query, long query_size, int verbose, long long* location, double* distance, DTWSettings *settings)
