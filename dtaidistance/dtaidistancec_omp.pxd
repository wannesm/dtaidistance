
from dtaidistancec cimport dtwvalue, DTWBlock, DTWSettings

cdef extern from "lib/DTAIDistanceC/DTAIDistanceC/dtw_openmp.h":
    size_t dtw_distances_ptrs_parallel(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths,
                                       dtwvalue* output, DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_ndim_ptrs_parallel(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths, int ndim,
                                            dtwvalue* output, DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_matrix_parallel(dtwvalue *matrix, size_t nb_rows, size_t nb_cols,
                                         dtwvalue* output, DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_ndim_matrix_parallel(dtwvalue *matrix, size_t nb_rows, size_t nb_cols, int ndim,
                                              dtwvalue* output, DTWBlock* block, DTWSettings* settings)
