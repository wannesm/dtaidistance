
from dtaidistancec_globals cimport seq_t, StepType, DDPath
from dtaidistancec_dtw cimport DTWSettings

cdef extern from "dd_loco.h":
    ctypedef struct LoCoSettings:
        Py_ssize_t window
        seq_t penalty
        Py_ssize_t psi_1b
        Py_ssize_t psi_2b
        bint only_triu
        seq_t gamma
        seq_t tau
        seq_t delta
        seq_t delta_factor
        StepType step_type

    LoCoSettings loco_settings_default()

    seq_t loco_warping_paths(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, LoCoSettings *settings)
    seq_t loco_warping_paths_typeI(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2,  LoCoSettings *settings)
    seq_t loco_warping_paths_ndim_typeI(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, int ndim,
                                        LoCoSettings *settings)
    seq_t loco_warping_paths_typeIII(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, LoCoSettings *settings)
    seq_t loco_warping_paths_ndim_typeIII(seq_t *wps, seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, int ndim,
                                          LoCoSettings *settings)
    void loco_path_negativize(Py_ssize_t *path, Py_ssize_t length, seq_t *wps, Py_ssize_t l1, Py_ssize_t l2, int buffer, int inf_rows,
                              int inf_cols)

    void best_path_init(DDPath *a, size_t initialSize)
    void best_path_insert(DDPath *a, Py_ssize_t element)
    void best_path_free(DDPath *a)
    DDPath loco_best_path(seq_t *wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t r, Py_ssize_t c, int min_size, LoCoSettings *settings)

    void loco_wps_argmax(seq_t *wps, Py_ssize_t l, Py_ssize_t *idxs, int n)
