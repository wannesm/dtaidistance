
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

    ctypedef struct DTWBlock:
        Py_ssize_t rb
        Py_ssize_t re
        Py_ssize_t cb
        Py_ssize_t ce
        bint triu

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
                               bint return_dtw, bint do_sqrt, bint psi_neg, DTWSettings *settings)
    seq_t dtw_warping_paths_ndim(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, int l2,
                                 bint return_dtw, bint do_sqrt, bint psi_neg, int ndim, DTWSettings *settings)
    void dtw_expand_wps(seq_t *wps, seq_t *full, Py_ssize_t l1, Py_ssize_t l2, DTWSettings *settings)
    Py_ssize_t dtw_best_path(seq_t *wps, Py_ssize_t *i1, Py_ssize_t *i2, Py_ssize_t l1, Py_ssize_t l2,
                             DTWSettings *settings)
    Py_ssize_t dtw_best_path_prob(seq_t *wps, Py_ssize_t *i1, Py_ssize_t *i2, Py_ssize_t l1, Py_ssize_t l2,
                                  seq_t avg, DTWSettings *settings);
    Py_ssize_t warping_path(seq_t *from_s, Py_ssize_t from_l, seq_t* to_s, Py_ssize_t to_l,
                            Py_ssize_t *from_i, Py_ssize_t *to_i, DTWSettings * settings)
    Py_ssize_t warping_path_ndim(seq_t *from_s, Py_ssize_t from_l, seq_t * to_s, Py_ssize_t to_l,
                                 Py_ssize_t *from_i, Py_ssize_t *to_i, int ndim, DTWSettings * settings)
    void dtw_srand(unsigned int seed)
    Py_ssize_t warping_path_prob_ndim(seq_t *from_s, Py_ssize_t from_l, seq_t* to_s, Py_ssize_t to_l,
                                      Py_ssize_t *from_i, Py_ssize_t *to_i, seq_t avg, int ndim, DTWSettings * settings)

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
    Py_ssize_t dtw_distances_length(DTWBlock *block, Py_ssize_t nb_series)
