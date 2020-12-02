from dtaidistancec_globals cimport seq_t
from dtaidistancec_dtw cimport DTWSettings

cdef extern from "dd_dtw_search.h":

    void lower_upper_naive(double *t, int len, double *l, double *u, DTWSettings *settings) nogil
    void lower_upper_lemire(double *t, int len, double *l, double *u, DTWSettings *settings) nogil
    void lower_upper_ascending_minima(double *t, int len, double *l, double *u, DTWSettings *settings) nogil
    double lb_keogh_from_envelope(double *s1, int l1, double *l, double *u, DTWSettings *settings) nogil
    int nn_lb_keogh(seq_t *data, Py_ssize_t data_size, seq_t *query, seq_t *lb, Py_ssize_t query_size, int verbose, Py_ssize_t *location, seq_t *distance, DTWSettings *settings) nogil
    int nn_lb_keogh_subsequence(seq_t *data, Py_ssize_t data_size, seq_t *query, seq_t *l, seq_t *u, Py_ssize_t query_size, int verbose, Py_ssize_t *location, seq_t *distance, DTWSettings *settings) nogil
    int ucrdtw(double* data, long long data_size, int skip, double* query, long query_size, int verbose, Py_ssize_t* location, double* distance, DTWSettings *settings) nogil
