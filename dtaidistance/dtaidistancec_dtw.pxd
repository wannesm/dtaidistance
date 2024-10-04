
from dtaidistancec_globals cimport seq_t


cdef extern from "dd_dtw.h":
    ctypedef struct DTWSettings:
        Py_ssize_t window
        seq_t max_dist
        seq_t max_step
        Py_ssize_t max_length_diff
        seq_t penalty
        Py_ssize_t psi_1b
        Py_ssize_t psi_1e
        Py_ssize_t psi_2b
        Py_ssize_t psi_2e
        bint use_pruning
        bint only_ub
        int inner_dist

    ctypedef struct DTWBlock:
        Py_ssize_t rb
        Py_ssize_t re
        Py_ssize_t cb
        Py_ssize_t ce
        bint triu

    ctypedef struct DTWWps:
        Py_ssize_t ldiff
        Py_ssize_t ldiffr
        Py_ssize_t ldiffc
        Py_ssize_t window
        Py_ssize_t width
        Py_ssize_t length
        Py_ssize_t ri1
        Py_ssize_t ri2
        Py_ssize_t ri3
        Py_ssize_t overlap_left_ri
        Py_ssize_t overlap_right_ri
        seq_t max_step
        seq_t max_dist
        seq_t penalty

    DTWSettings dtw_settings_default()
    Py_ssize_t dtw_settings_wps_length(Py_ssize_t l1, Py_ssize_t l2, DTWSettings *settings)
    Py_ssize_t dtw_settings_wps_width(Py_ssize_t l1, Py_ssize_t l2, DTWSettings *settings)
    void dtw_settings_set_psi(Py_ssize_t psi, DTWSettings *settings)
    void dtw_print_settings(DTWSettings *settings)

    seq_t dtw_distance(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2,
                          DTWSettings *settings)
    seq_t dtw_distance_ndim(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, int ndim,
                               DTWSettings *settings)

    seq_t dtw_warping_paths(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, int l2,
                               bint return_dtw, bint keep_int_repr, bint psi_neg, DTWSettings *settings)
    seq_t dtw_warping_paths_ndim(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, int l2,
                                 bint return_dtw, bint keep_int_repr, bint psi_neg, int ndim, DTWSettings *settings)
    seq_t dtw_warping_paths_affinity(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, bint return_dtw,
                                     bint keep_int_repr, bint psi_neg, bint only_triu, seq_t gamma, seq_t tau, seq_t delta,
                                     seq_t delta_factor, DTWSettings *settings)
    seq_t dtw_warping_paths_affinity_ndim(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, bint return_dtw,
                                          bint keep_int_repr, bint psi_neg, bint only_triu, int ndim, seq_t gamma, seq_t tau, seq_t delta,
                                          seq_t delta_factor, DTWSettings *settings)
    void dtw_expand_wps(seq_t *wps, seq_t *full, Py_ssize_t l1, Py_ssize_t l2, DTWSettings *settings)
    void dtw_expand_wps_slice(seq_t *wps, seq_t *full, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t rb, Py_ssize_t re, Py_ssize_t cb, Py_ssize_t ce,
                              DTWSettings *settings)
    void dtw_expand_wps_affinity(seq_t *wps, seq_t *full, Py_ssize_t l1, Py_ssize_t l2, DTWSettings *settings)
    void dtw_expand_wps_slice_affinity(seq_t *wps, seq_t *full, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t rb, Py_ssize_t re, Py_ssize_t cb,
                                       Py_ssize_t ce, DTWSettings *settings)
    void dtw_wps_negativize_value(DTWWps *p, seq_t *wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t r, Py_ssize_t c)
    void dtw_wps_positivize_value(DTWWps *p, seq_t *wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t r, Py_ssize_t c)
    void dtw_wps_positivize(DTWWps *p, seq_t *wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t rb, Py_ssize_t re, Py_ssize_t cb, Py_ssize_t ce, bint intersection)
    void dtw_wps_negativize(DTWWps *p, seq_t *wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t rb, Py_ssize_t re, Py_ssize_t cb, Py_ssize_t ce, bint intersection)
    Py_ssize_t dtw_wps_loc(DTWWps *p, Py_ssize_t r, Py_ssize_t c, Py_ssize_t l1, Py_ssize_t l2)
    Py_ssize_t dtw_wps_max(DTWWps * p, seq_t *wps, Py_ssize_t *r, Py_ssize_t *c, Py_ssize_t l1, Py_ssize_t l2)
    Py_ssize_t dtw_best_path(seq_t *wps, Py_ssize_t *i1, Py_ssize_t *i2, Py_ssize_t l1, Py_ssize_t l2,
                             DTWSettings *settings)
    Py_ssize_t dtw_best_path_affinity(seq_t *wps, Py_ssize_t *i1, Py_ssize_t *i2, Py_ssize_t l1, Py_ssize_t l2,
                                      Py_ssize_t s1, Py_ssize_t s2,
                                      DTWSettings *settings)
    Py_ssize_t dtw_best_path_prob(seq_t *wps, Py_ssize_t *i1, Py_ssize_t *i2, Py_ssize_t l1, Py_ssize_t l2,
                                  seq_t avg, DTWSettings *settings);
    seq_t dtw_warping_path(seq_t *from_s, Py_ssize_t from_l, seq_t* to_s, Py_ssize_t to_l,
                           Py_ssize_t *from_i, Py_ssize_t *to_i, Py_ssize_t *length_i, DTWSettings * settings)
    seq_t dtw_warping_path_ndim(seq_t *from_s, Py_ssize_t from_l, seq_t * to_s, Py_ssize_t to_l,
                                Py_ssize_t *from_i, Py_ssize_t *to_i, Py_ssize_t *length_i, int ndim, DTWSettings * settings)
    void dtw_srand(unsigned int seed)
    seq_t dtw_warping_path_prob_ndim(seq_t *from_s, Py_ssize_t from_l, seq_t* to_s, Py_ssize_t to_l,
                                     Py_ssize_t *from_i, Py_ssize_t *to_i, Py_ssize_t *length_i, seq_t avg, int ndim, DTWSettings * settings)
    DTWWps dtw_wps_parts(Py_ssize_t l1, Py_ssize_t l2, DTWSettings * settings)

    seq_t ub_euclidean(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2)
    seq_t ub_euclidean_ndim(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, int ndim)
    seq_t lb_keogh(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, DTWSettings *settings)

    void dtw_dba_ptrs(seq_t **ptrs, Py_ssize_t nb_ptrs, Py_ssize_t* lengths,
                  seq_t *c, Py_ssize_t t, unsigned char *mask, int prob_samples, int ndim, DTWSettings *settings)
    void dtw_dba_matrix(seq_t *matrix, Py_ssize_t nb_rows, Py_ssize_t nb_cols,
                    seq_t *c, Py_ssize_t t, unsigned char *mask, int prob_samples, int ndim, DTWSettings *settings)

    DTWBlock dtw_empty_block()
    Py_ssize_t dtw_distances_ptrs(seq_t **ptrs, Py_ssize_t nb_ptrs, Py_ssize_t* lengths, seq_t* output,
                          DTWBlock* block, DTWSettings* settings)
    Py_ssize_t dtw_distances_ndim_ptrs(seq_t **ptrs, Py_ssize_t nb_ptrs, Py_ssize_t* lengths, int ndim, seq_t* output,
                                   DTWBlock* block, DTWSettings* settings)
    Py_ssize_t dtw_distances_matrix(seq_t *matrix, Py_ssize_t nb_rows, Py_ssize_t nb_cols, seq_t* output,
                            DTWBlock* block, DTWSettings* settings)
    Py_ssize_t dtw_distances_ndim_matrix(seq_t *matrix, Py_ssize_t nb_rows, Py_ssize_t nb_cols, int ndim, seq_t* output,
                                     DTWBlock* block, DTWSettings* settings)
    Py_ssize_t dtw_distances_matrices(seq_t *matrix_r, Py_ssize_t nb_rows_r, Py_ssize_t nb_cols_r,
                                      seq_t *matrix_c, Py_ssize_t nb_rows_c, Py_ssize_t nb_cols_c,
                                      seq_t * output, DTWBlock * block, DTWSettings * settings)
    Py_ssize_t dtw_distances_length(DTWBlock *block, Py_ssize_t nb_series_r, Py_ssize_t nb_series_c)

    void dtw_print_wps(seq_t * wps, Py_ssize_t l1, Py_ssize_t l2, DTWSettings * settings)
    void dtw_print_wps_compact(seq_t * wps, Py_ssize_t l1, Py_ssize_t l2, DTWSettings * settings)
