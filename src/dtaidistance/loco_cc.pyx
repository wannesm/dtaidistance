"""
dtaidistance.loco_cc
~~~~~~~~~~~~~~~~~~~~

Local Concurrences, C implementation.

:author: Wannes Meert
:copyright: Copyright 2023 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
cimport dtaidistancec_loco
from dtaidistancec_dtw cimport seq_t
from cython.view cimport array as cvarray
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf, fseek, ftell, SEEK_END, rewind, fread

logger = logging.getLogger("be.kuleuven.dtai.distance")


def save_wp(seq_t[:, :] wps, filename):
    cdef bytes py_bytes = filename.encode()
    cdef char * c_string = py_bytes
    cdef Py_ssize_t N = wps.shape[0] * wps.shape[1]
    cdef FILE *ptr_fw
    cdef seq_t *ptr_d = &wps[0,0]
    ptr_fw = fopen(c_string, "wb")
    fwrite(ptr_d, sizeof(seq_t), N, ptr_fw)
    fclose(ptr_fw)


cdef class LoCoSettings:
    def __cinit__(self):
        pass

    def __init__(self, **kwargs):
        self._settings = dtaidistancec_loco.loco_settings_default()

        self._settings.window = 0
        if "window" in kwargs:
            if kwargs["window"] is None:
                pass
            else:
                self._settings.window = kwargs["window"]

        self._settings.penalty = 0
        if "penalty" in kwargs:
            if kwargs["penalty"] is None:
                pass
            else:
                self._settings.penalty = kwargs["penalty"]

        self._settings.psi_1b = self._settings.psi_2b = 0
        if "psi" in kwargs:
            if kwargs["psi"] is None:
                pass
            elif type(kwargs["psi"]) is int:
                self._settings.psi_1b = self._settings.psi_2b = kwargs["psi"]
            elif type(kwargs["psi"]) is tuple or type(kwargs["psi"]) is list:
                self._settings.psi_1b, self._settings.psi_2b = kwargs["psi"]

        self._settings.only_triu = kwargs.get("only_triu", 0)
        self._settings.gamma = kwargs.get("gamma", 0)
        self._settings.tau = kwargs.get("tau", 0)
        self._settings.delta = kwargs.get("delta", 0)
        self._settings.delta_factor = kwargs.get("delta_factor", 0)

        if "step_type" in kwargs:
            step_type = kwargs["step_type"]
            if step_type == "TypeI":
                self._settings.step_type = dtaidistancec_loco.TypeI
            elif step_type == "TypeIII":
                self._settings.step_type = dtaidistancec_loco.TypeIII
            else:
                raise AttributeError("Unknown step_type: {}".format(step_type))

    @property
    def step_type(self):
        return self._settings.step_type

    def inf_rowscols(self):
        if self._settings.step_type == dtaidistancec_loco.TypeI:
            return 1, 1
        elif self._settings.step_type == dtaidistancec_loco.TypeIII:
            return 2, 2
        else:
            raise AttributeError("Unknown step_type: {}".format(self._settings.step_type))


def loco_warping_paths(seq_t[:, :] wps, seq_t[:] s1, seq_t[:] s2, ndim=1, **kwargs):
    settings = LoCoSettings(**kwargs)
    if settings.step_type == dtaidistancec_loco.TypeI:
        settings._settings.step_type = dtaidistancec_loco.TypeI
        dtaidistancec_loco.loco_warping_paths_ndim_typeI(&wps[0,0], &s1[0], len(s1), &s2[0], len(s2),
                                                         ndim, &settings._settings)
    elif settings.step_type == dtaidistancec_loco.TypeIII:
        settings._settings.step_type = dtaidistancec_loco.TypeIII
        dtaidistancec_loco.loco_warping_paths_ndim_typeIII(&wps[0,0], &s1[0], len(s1), &s2[0], len(s2),
                                                           ndim, &settings._settings)
    else:
        raise ValueError("Unknown steps type.")

def loco_best_path(seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t r, Py_ssize_t c, int min_size, **kwargs):
    settings = LoCoSettings(**kwargs)
    inf_rows, inf_cols = settings.inf_rowscols()
    # print(f"{inf_rows=}, {inf_cols=}, {r=}, {c=}")
    cdef Py_ssize_t width = l2 + inf_cols
    cdef dtaidistancec_loco.BestPath bp = dtaidistancec_loco.loco_best_path(&wps[0,0], l1, l2, r, c, min_size,
                                                                            &settings._settings)
    if bp.used == 0:
        print('ERROR bp.used = 0')
        return None
    try:
        # Use cython.view.array to avoid numpy dependency
        idxs = cvarray(shape=(bp.used, 2), itemsize=sizeof(Py_ssize_t), format="l")
    except MemoryError as exc:
        dtaidistancec_loco.best_path_free(&bp)
        print("Cannot allocate memory for warping paths matrix.")
        raise exc
    for i in range(0, bp.used):
        idxs[i, 0] = bp.array[bp.used - i - 1] // width - inf_rows
        idxs[i, 1] = bp.array[bp.used - i - 1] % width - inf_cols
    dtaidistancec_loco.best_path_free(&bp)
    return idxs

def loco_path_negativize(Py_ssize_t[:, :] path, seq_t[:, :] wps, int buffer, int inf_rows,
                              int inf_cols):
    dtaidistancec_loco.loco_path_negativize(&path[0,0], path.shape[0], &wps[0,0], wps.shape[0]-inf_rows, wps.shape[1]-inf_cols, buffer, inf_rows, inf_cols)

def loco_wps_argmax(seq_t[:, :] wps, Py_ssize_t[:] idxs, int n):
    l = wps.shape[0] * wps.shape[1]
    dtaidistancec_loco.loco_wps_argmax(&wps[0,0], l, &idxs[0], n)
