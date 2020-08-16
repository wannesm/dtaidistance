
from dtaidistancec_globals cimport seq_t


cdef extern from "dd_dtw.h":
    ctypedef struct DTWSettings:
        size_t window
        seq_t max_dist
        seq_t max_step
        size_t max_length_diff
        seq_t penalty
        size_t psi
        bint use_ssize_t
        bint use_pruning
        bint only_ub

    ctypedef struct DTWBlock:
        size_t rb
        size_t re
        size_t cb
        size_t ce

    DTWSettings dtw_settings_default()
    void dtw_print_settings(DTWSettings *settings)

    seq_t dtw_distance(seq_t *s1, size_t l1, seq_t *s2, size_t l2,
                          DTWSettings *settings)
    seq_t dtw_distance_ndim(seq_t *s1, size_t l1, seq_t *s2, size_t l2, int ndim,
                               DTWSettings *settings)
    seq_t dtw_warping_paths(seq_t *wps, seq_t *s1, size_t l1, seq_t *s2, int l2,
                               bint return_dtw, bint do_sqrt, DTWSettings *settings)

    seq_t ub_euclidean(seq_t *s1, size_t l1, seq_t *s2, size_t l2)
    seq_t ub_euclidean_ndim(seq_t *s1, size_t l1, seq_t *s2, size_t l2, int ndim)
    seq_t lb_keogh(seq_t *s1, size_t l1, seq_t *s2, size_t l2, DTWSettings *settings)

    DTWBlock dtw_empty_block()
    size_t dtw_distances_ptrs(seq_t **ptrs, size_t nb_ptrs, size_t* lengths, seq_t* output,
                          DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_ndim_ptrs(seq_t **ptrs, size_t nb_ptrs, size_t* lengths, int ndim, seq_t* output,
                                   DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_matrix(seq_t *matrix, size_t nb_rows, size_t nb_cols, seq_t* output,
                            DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_ndim_matrix(seq_t *matrix, size_t nb_rows, size_t nb_cols, int ndim, seq_t* output,
                                     DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_length(DTWBlock *block, size_t nb_series, bint use_ssize_t)
