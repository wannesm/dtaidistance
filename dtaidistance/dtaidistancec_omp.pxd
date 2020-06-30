
from dtaidistancec cimport dtwvalue, DTWBlock, DTWSettings

cdef extern from "lib/DTAIDistanceC/DTAIDistanceC/dtw_openmp.h":
    size_t dtw_distances_ptrs_parallel(dtwvalue **ptrs, int nb_ptrs, int* lengths,
                                       dtwvalue* output, DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_matrix_parallel(dtwvalue *matrix, int nb_rows, int nb_cols,
                                         dtwvalue* output, DTWBlock* block, DTWSettings* settings)
