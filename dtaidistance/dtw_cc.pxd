cimport dtaidistancec_dtw

cdef class DTWBlock:
    cdef dtaidistancec_dtw.DTWBlock _block

cdef class DTWSettings:
    cdef dtaidistancec_dtw.DTWSettings _settings

cdef class DTWSeriesPointers:
    cdef double **_ptrs
    cdef Py_ssize_t *_lengths
    cdef Py_ssize_t _nb_ptrs

cdef class DTWSeriesMatrix:
    cdef double[:,::1] _data

cdef class DTWSeriesMatrixNDim:
    cdef double[:,:,::1] _data
