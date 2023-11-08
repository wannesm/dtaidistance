
from dtaidistancec_globals cimport seq_t


cdef extern from "dd_ed.h":
    seq_t euclidean_distance(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2)
    seq_t euclidean_distance_euclidean(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2)
    seq_t euclidean_distance_ndim(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, int ndim)
    seq_t euclidean_distance_ndim_euclidean(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, int ndim)
