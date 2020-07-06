

cdef extern from "lib/DTAIDistanceC/DTAIDistanceC/dtw.h":
    ctypedef double dtwvalue
    ctypedef struct DTWSettings:
        size_t window
        dtwvalue max_dist
        dtwvalue max_step
        size_t max_length_diff
        dtwvalue penalty
        size_t psi
        bint use_ssize_t
        bint use_pruning

    ctypedef struct DTWBlock:
        size_t rb
        size_t re
        size_t cb
        size_t ce

    DTWSettings dtw_settings_default()
    void dtw_print_settings(DTWSettings *settings)

    dtwvalue dtw_distance(dtwvalue *s1, size_t l1, dtwvalue *s2, size_t l2,
                          DTWSettings *settings)
    dtwvalue dtw_warping_paths(dtwvalue *wps, dtwvalue *s1, size_t l1, dtwvalue *s2, int l2,
                               bint return_dtw, bint do_sqrt, DTWSettings *settings)

    dtwvalue ub_euclidean(dtwvalue *s1, size_t l1, dtwvalue *s2, size_t l2)
    dtwvalue lb_keogh(dtwvalue *s1, size_t l1, dtwvalue *s2, size_t l2, DTWSettings *settings)

    DTWBlock dtw_empty_block()
    size_t dtw_distances_ptrs(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths, dtwvalue* output,
                          DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_matrix(dtwvalue *matrix, size_t nb_rows, size_t nb_cols, dtwvalue* output,
                            DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_length(DTWBlock *block, size_t nb_series, bint use_ssize_t)
