from dtaidistancec_globals cimport seq_t
from dtaidistancec_dtw cimport DTWSettings

cdef extern from "dd_dtw_search.h":

    void lower_upper_naive(double *t, int len, double *l, double *u, DTWSettings *settings) nogil
    void lower_upper_lemire(double *t, int len, double *l, double *u, DTWSettings *settings) nogil
    void lower_upper_ascending_minima(double *t, int len, double *l, double *u, DTWSettings *settings) nogil
    double lb_keogh_from_envelope(double *s1, int l1, double *l, double *u, DTWSettings *settings) nogil
    int nn_lb_keogh(seq_t *data, Py_ssize_t data_size, int skip, seq_t *query, seq_t *L, seq_t *U, Py_ssize_t query_size, int verbose, Py_ssize_t *location, seq_t *distance, DTWSettings *settings)
    int ucrdtw(double* data, long long data_size, int skip, double* query, long query_size, int verbose, long long* location, double* distance, DTWSettings *settings)
