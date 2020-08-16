
from dtaidistancec_globals cimport seq_t


cdef extern from "dd_ed.h":
    seq_t euclidean_distance(seq_t *s1, size_t l1, seq_t *s2, size_t l2)
    seq_t euclidean_distance_ndim(seq_t *s1, size_t l1, seq_t *s2, size_t l2, int ndim)
