

cdef extern from "lib/DTAIDistanceC/DTAIDistanceC/dtw.h":
    ctypedef double dtwvalue
    ctypedef struct DTWSettings:
        int window
        dtwvalue max_dist
        dtwvalue max_step
        int max_length_diff
        dtwvalue penalty
        int psi

    ctypedef struct DTWBlock:
        int rb
        int re
        int cb
        int ce

    DTWSettings dtw_default_settings()
    dtwvalue dtw_distance(dtwvalue *s1, int l1, dtwvalue *s2, int l2,
                          DTWSettings *settings)
    dtwvalue dtw_warping_paths(dtwvalue *wps, dtwvalue *s1, int l1, dtwvalue *s2, int l2,
                               bint return_dtw, bint do_sqrt, DTWSettings *settings)

    void dtw_print_settings(DTWSettings *settings)

    DTWBlock dtw_empty_block()
    size_t dtw_distances_ptrs(dtwvalue **ptrs, int nb_ptrs, int* lengths, dtwvalue* output,
                          DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_matrix(dtwvalue *matrix, int nb_rows, int nb_cols, dtwvalue* output,
                            DTWBlock* block, DTWSettings* settings)
    size_t dtw_distances_length(DTWBlock *block, int nb_series)
