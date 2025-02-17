
from dtaidistancec_globals cimport seq_t
from dtaidistancec_dtw cimport DTWBlock, DTWSettings


cdef extern from "dd_dtw_openmp.h":
    bint is_openmp_supported()
    Py_ssize_t dtw_distances_ptrs_parallel(seq_t **ptrs, Py_ssize_t nb_ptrs, Py_ssize_t* lengths,
                                           seq_t* output, DTWBlock* block, DTWSettings* settings)
    Py_ssize_t dtw_distances_ndim_ptrs_parallel(seq_t **ptrs, Py_ssize_t nb_ptrs, Py_ssize_t* lengths, int ndim,
                                                seq_t* output, DTWBlock* block, DTWSettings* settings)
    Py_ssize_t dtw_distances_matrix_parallel(seq_t *matrix, Py_ssize_t nb_rows, Py_ssize_t nb_cols,
                                             seq_t* output, DTWBlock* block, DTWSettings* settings)
    Py_ssize_t dtw_distances_ndim_matrix_parallel(seq_t *matrix, Py_ssize_t nb_rows, Py_ssize_t nb_cols, int ndim,
                                                  seq_t* output, DTWBlock* block, DTWSettings* settings)
    Py_ssize_t dtw_distances_matrices_parallel(seq_t *matrix_r, Py_ssize_t nb_rows_r, Py_ssize_t nb_cols_r,
                                               seq_t *matrix_c, Py_ssize_t nb_rows_c, Py_ssize_t nb_cols_c,
                                               seq_t * output, DTWBlock * block, DTWSettings * settings)
    Py_ssize_t dtw_distances_ndim_matrices_parallel(seq_t *matrix_r, Py_ssize_t nb_rows_r, Py_ssize_t nb_cols_r,
                                                    seq_t *matrix_c, Py_ssize_t nb_rows_c, Py_ssize_t nb_cols_c, int ndim,
                                                    seq_t * output, DTWBlock * block, DTWSettings * settings)
