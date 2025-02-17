cimport dtaidistancec_dtw
from dtaidistancec_dtw cimport seq_t

cdef class DTWBlock:
    cdef dtaidistancec_dtw.DTWBlock _block

cdef class DTWSettings:
    cdef dtaidistancec_dtw.DTWSettings _settings

cdef class DTWWps:
    cdef dtaidistancec_dtw.DTWWps _wps

cdef class DTWSeriesPointers:
    cdef seq_t **_ptrs
    cdef Py_ssize_t *_lengths
    cdef Py_ssize_t _nb_ptrs

cdef class DTWSeriesMatrix:
    cdef seq_t[:,::1] _data

cdef class DTWSeriesMatrixNDim:
    cdef seq_t[:,:,::1] _data
