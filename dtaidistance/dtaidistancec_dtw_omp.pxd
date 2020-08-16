
from dtaidistancec_globals cimport seq_t
from dtaidistancec_dtw cimport DTWBlock, DTWSettings


cdef extern from "dd_dtw_openmp.h":
    size_t dtw_distances_ptrs_parallel(seq_t **ptrs, size_t nb_ptrs, size_t* lengths,
                                       seq_t* output, DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_ndim_ptrs_parallel(seq_t **ptrs, size_t nb_ptrs, size_t* lengths, int ndim,
                                            seq_t* output, DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_matrix_parallel(seq_t *matrix, size_t nb_rows, size_t nb_cols,
                                         seq_t* output, DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_ndim_matrix_parallel(seq_t *matrix, size_t nb_rows, size_t nb_cols, int ndim,
                                              seq_t* output, DTWBlock* block, DTWSettings* settings)
