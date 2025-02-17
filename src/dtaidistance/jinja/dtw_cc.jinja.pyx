"""
dtaidistance.dtw_cc
~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW), C implementation.

:author: Wannes Meert
:copyright: Copyright 2017-2022 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
from cpython cimport array
import array
from cython import Py_ssize_t
from cython.view cimport array as cvarray
from libc.stdlib cimport abort, malloc, free, abs, labs
from libc.stdint cimport intptr_t
from libc.stdio cimport printf
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cimport dtaidistancec_dtw
cimport dtaidistancec_globals
from dtaidistancec_dtw cimport seq_t


cdef extern from "Python.h":
    Py_ssize_t PY_SSIZE_T_MAX


cdef class DTWBlock:
    def __cinit__(self):
        pass

    def __init__(self, rb, re, cb, ce, triu=True):
        self._block.rb = rb
        self._block.re = re
        self._block.cb = cb
        self._block.ce = ce
        self._block.triu = triu

    @property
    def rb(self):
        return self._block.rb

    @property
    def re(self):
        return self._block.re

    def re_set(self, value):
        self._block.re = value

    @property
    def cb(self):
        return self._block.cb

    @property
    def ce(self):
        return self._block.ce

    def ce_set(self, value):
        self._block.ce = value

    @property
    def triu(self):
        return self._block.triu

    def triu_set(self, value):
        self._block.triu = value

    def __str__(self):
        return f'DTWBlock(rb={self.rb},re={self.re},cb={self.cb},ce={self.ce},triu={self.triu})'


cdef class DTWWps:
    def __cinit__(self):
        pass

    def __init__(self, l1, l2, DTWSettings settings):
        self._wps = dtaidistancec_dtw.dtw_wps_parts(l1, l2, &settings._settings)

    @property
    def ri1(self):
        return self._wps.ri1

    @property
    def ri2(self):
        return self._wps.ri2

    @property
    def ri3(self):
        return self._wps.ri3


cdef class DTWSettings:
    def __cinit__(self):
        pass

    def __init__(self, **kwargs):
        self._settings = dtaidistancec_dtw.dtw_settings_default()
        if "window" in kwargs:
            if kwargs["window"] is None:
                self._settings.window = 0
            else:
                self._settings.window = kwargs["window"]
        if "max_dist" in kwargs:
            if kwargs["max_dist"] is None:
                self._settings.max_dist = 0
            else:
                self._settings.max_dist = kwargs["max_dist"]
        if "max_step" in kwargs:
            if kwargs["max_step"] is None:
                self._settings.max_step = 0
            else:
                self._settings.max_step = kwargs["max_step"]
        if "max_length_diff" in kwargs:
            if kwargs["max_length_diff"] is None:
                self._settings.max_length_diff = 0
            else:
                self._settings.max_length_diff = kwargs["max_length_diff"]
        if "penalty" in kwargs:
            if kwargs["penalty"] is None:
                self._settings.penalty = 0
            else:
                self._settings.penalty = kwargs["penalty"]
        if "psi" in kwargs:
            if kwargs["psi"] is None:
                self._settings.psi_1b = 0
                self._settings.psi_1e = 0
                self._settings.psi_2b = 0
                self._settings.psi_2e = 0
            else:
                if type(kwargs["psi"]) is int:
                    self._settings.psi_1b = kwargs["psi"]
                    self._settings.psi_1e = kwargs["psi"]
                    self._settings.psi_2b = kwargs["psi"]
                    self._settings.psi_2e = kwargs["psi"]
                elif type(kwargs["psi"]) is tuple or type(kwargs["psi"]) is list:
                    if len(kwargs["psi"]) != 4:
                        self._settings.psi_1b = 0
                        self._settings.psi_1e = 0
                        self._settings.psi_2b = 0
                        self._settings.psi_2e = 0
                    else:
                        self._settings.psi_1b = kwargs["psi"][0]
                        self._settings.psi_1e = kwargs["psi"][1]
                        self._settings.psi_2b = kwargs["psi"][2]
                        self._settings.psi_2e = kwargs["psi"][3]
        if "use_pruning" in kwargs:
            if kwargs["use_pruning"] is None:
                self._settings.use_pruning = False
            else:
                self._settings.use_pruning = kwargs["use_pruning"]
        if "only_ub" in kwargs:
            if kwargs["only_ub"] is None:
                self._settings.only_ub = False
            else:
                self._settings.only_ub = kwargs["only_ub"]
        if "inner_dist" in kwargs:
            if kwargs["inner_dist"] == "squared euclidean" or kwargs["inner_dist"] == 0:
                self._settings.inner_dist = 0
            elif kwargs["inner_dist"] == "euclidean" or kwargs["inner_dist"] == 1:
                self._settings.inner_dist = 1
            else:
                raise AttributeError("Unknown inner_dist: {}".format(kwargs["inner_dist"]))

    @property
    def window(self):
        return self._settings.window

    @property
    def max_dist(self):
        return self._settings.max_dist

    @property
    def max_step(self):
        return self._settings.max_step

    @property
    def max_length_diff(self):
        return self._settings.max_length_diff

    @property
    def penalty(self):
        return self._settings.penalty

    @property
    def psi(self):
        return {
            '1b': self._settings.psi_1b,
            '1e': self._settings.psi_1e,
            '2b': self._settings.psi_2b,
            '2e': self._settings.psi_2e,
        }

    @property
    def use_pruning(self):
        return self._settings.use_pruning

    @property
    def only_ub(self):
        return self._settings.only_ub

    @property
    def inner_dist(self):
        if self._settings.inner_dist == 0:
            return "squared euclidean"
        elif self._settings.inner_dist == 1:
            return "euclidean"
        else:
            return "unknown inner distance"

    def __str__(self):
        return (
            "DTWSettings {\n"
            f"  window = {self.window}\n"
            f"  max_dist = {self.max_dist}\n"
            f"  max_step = {self.max_step}\n"
            f"  max_length_diff = {self.max_length_diff}\n"
            f"  penalty = {self.penalty}\n"
            f"  psi = {self.psi}\n"
            f"  use_pruning = {self.use_pruning}\n"
            f"  only_ub = {self.only_ub}\n"
            f"  inner_dist = {self.inner_dist}\n"
            "}")


cdef class DTWSeriesPointers:
    def __cinit__(self, int nb_series):
        self._ptrs = <seq_t **> malloc(nb_series * sizeof(seq_t*))
        self._nb_ptrs = nb_series
        if not self._ptrs:
            self._ptrs = NULL
            raise MemoryError()
        self._lengths = <Py_ssize_t *> malloc(nb_series * sizeof(Py_ssize_t))
        if not self._lengths:
            self._lengths = NULL
            raise MemoryError()

    def __dealloc__(self):
        if self._ptrs is not NULL:
            free(self._ptrs)
        if self._lengths is not NULL:
            free(self._lengths)


cdef class DTWSeriesMatrix:
    def __cinit__(self, seq_t[:, ::1] data):
        self._data = data

    @property
    def nb_rows(self):
        return self._data.shape[0]

    @property
    def nb_cols(self):
        return self._data.shape[1]


cdef class DTWSeriesMatrixNDim:
    def __cinit__(self, seq_t[:, :, ::1] data):
        self._data = data

    @property
    def nb_rows(self):
        return self._data.shape[0]

    @property
    def nb_cols(self):
        return self._data.shape[1]

    @property
    def nb_dims(self):
        return self._data.shape[2]


def dtw_series_from_data(data, force_pointers=False):
    cdef DTWSeriesPointers ptrs
    cdef DTWSeriesMatrix matrix
    cdef intptr_t ptr
    if force_pointers or isinstance(data, list) or isinstance(data, set) or isinstance(data, tuple):
        ptrs = DTWSeriesPointers(len(data))
        for i in range(len(data)):
            ptr = data[i].ctypes.data  # uniform for memoryviews and numpy
            ptrs._ptrs[i] = <seq_t *> ptr
            ptrs._lengths[i] = len(data[i])
        return ptrs
    try:
        matrix = DTWSeriesMatrix(data)
        return matrix
    except ValueError:
        pass
    try:
        matrix = DTWSeriesMatrixNDim(data)
        return matrix
    except ValueError:
        raise ValueError(f"Cannot convert data of type {type(data)}")


def ub_euclidean(seq_t[:] s1, seq_t[:] s2):
    """ See ed.euclidean_distance"""
    return dtaidistancec_dtw.ub_euclidean(&s1[0], len(s1), &s2[0], len(s2))


def ub_euclidean_ndim(seq_t[:, :] s1, seq_t[:, :] s2):
    """ See ed.euclidean_distance_ndim"""
    # Assumes C contiguous
    if s1.shape[1] != s2.shape[1]:
        raise Exception("Dimension of sequence entries needs to be the same: {} != {}".format(s1.shape[1], s2.shape[1]))
    ndim = s1.shape[1]
    return dtaidistancec_dtw.ub_euclidean_ndim(&s1[0,0], len(s1), &s2[0,0], len(s2), ndim)


def lb_keogh(seq_t[:] s1, seq_t[:] s2, **kwargs):
    # Assumes C contiguous
    settings = DTWSettings(**kwargs)
    return dtaidistancec_dtw.lb_keogh(&s1[0], len(s1), &s2[0], len(s2), &settings._settings)


def distance(seq_t[:] s1, seq_t[:] s2, **kwargs):
    """DTW distance.

    Assumes C-contiguous arrays.

    See distance().
    :param s1: First sequence (buffer of seq_t-s)
    :param s2: Second sequence (buffer of seq_t-s)
    :param kwargs: Settings (see DTWSettings)
    """
    # Assumes C contiguous
    settings = DTWSettings(**kwargs)
    return dtaidistancec_dtw.dtw_distance(&s1[0], len(s1), &s2[0], len(s2), &settings._settings)


def distance_ndim(seq_t[:, :] s1, seq_t[:, :] s2, **kwargs):
    """DTW distance for n-dimensional arrays.

    Assumes C-contiguous arrays.

    See distance().
    :param s1: First sequence (buffer of seq_t-s)
    :param s2: Second sequence (buffer of seq_t-s)
    :param ndim: Number of dimensions
    :param kwargs: Settings (see DTWSettings)
    """
    # Assumes C contiguous
    settings = DTWSettings(**kwargs)
    if s1.shape[1] != s2.shape[1]:
        raise Exception("Dimension of sequence entries needs to be the same: {} != {}".format(s1.shape[1], s2.shape[1]))
    ndim = s1.shape[1]
    return dtaidistancec_dtw.dtw_distance_ndim(&s1[0,0], len(s1), &s2[0,0], len(s2), ndim, &settings._settings)


def distance_ndim_assinglearray(seq_t[:] s1, seq_t[:] s2, int ndim, **kwargs):
    """DTW distance for n-dimensional arrays.

    Assumes C-contiguous arrays (with sequence item as first dimension).

    See distance().
    :param s1: First sequence (buffer of seq_ts)
    :param s2: Second sequence (buffer of seq_ts)
    :param ndim: Number of dimensions
    :param kwargs: Settings (see DTWSettings)
    """
    # Assumes C contiguous
    settings = DTWSettings(**kwargs)
    return dtaidistancec_dtw.dtw_distance_ndim(&s1[0], len(s1), &s2[0], len(s2), ndim, &settings._settings)


def wps_length(Py_ssize_t l1, Py_ssize_t l2, **kwargs):
    settings = DTWSettings(**kwargs)
    return dtaidistancec_dtw.dtw_settings_wps_length(l1, l2, &settings._settings)


def wps_width(Py_ssize_t l1, Py_ssize_t l2, **kwargs):
    settings = DTWSettings(**kwargs)
    return dtaidistancec_dtw.dtw_settings_wps_width(l1, l2, &settings._settings)

{% set suffix = '' %}
{%- include 'dtw_cc_warpingpaths.jinja.pyx' %}

{% set suffix = '_ndim' %}
{%- include 'dtw_cc_warpingpaths.jinja.pyx' %}

{% set suffix = '_affinity' %}
{%- include 'dtw_cc_warpingpaths.jinja.pyx' %}

{% set suffix = '' %}
{%- include 'dtw_cc_warpingpath.jinja.pyx' %}

{% set suffix = '_ndim' %}
{%- include 'dtw_cc_warpingpath.jinja.pyx' %}

def wps_negativize_value(DTWWps p, seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t r, Py_ssize_t c):
    dtaidistancec_dtw.dtw_wps_negativize_value(&p._wps, &wps[0,0], l1, l2, r, c)

def wps_positivize_value(DTWWps p, seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t r, Py_ssize_t c):
    dtaidistancec_dtw.dtw_wps_positivize_value(&p._wps, &wps[0,0], l1, l2, r, c)

def wps_negativize(DTWWps p, seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t rb, Py_ssize_t re, Py_ssize_t cb, Py_ssize_t ce, bint intersection):
    dtaidistancec_dtw.dtw_wps_negativize(&p._wps, &wps[0,0], l1, l2, rb, re, cb, ce, intersection)

def wps_positivize(DTWWps p, seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t rb, Py_ssize_t re, Py_ssize_t cb, Py_ssize_t ce, bint intersection):
    dtaidistancec_dtw.dtw_wps_positivize(&p._wps, &wps[0,0], l1, l2, rb, re, cb, ce, intersection)

def wps_max(DTWWps p, seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2):
    cdef Py_ssize_t r, c
    result = dtaidistancec_dtw.dtw_wps_max(&p._wps, &wps[0, 0], &r, &c, l1, l2)
    return r, c

def wps_expand_slice(seq_t[:, :] wps, seq_t[:, :] slice, Py_ssize_t l1, Py_ssize_t l2,
                     Py_ssize_t rb, Py_ssize_t re, Py_ssize_t cb, Py_ssize_t ce,
                     DTWSettings settings):
    dtaidistancec_dtw.dtw_expand_wps_slice_affinity(&wps[0, 0], &slice[0, 0],
                                                    l1, l2, rb, re, cb, ce,
                                                    &settings._settings)

def wps_print(seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2, **kwargs):
    settings = DTWSettings(**kwargs)
    dtaidistancec_dtw.dtw_print_wps(&wps[0,0], l1, l2, &settings._settings)

def wps_print_compact(seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2, **kwargs):
    settings = DTWSettings(**kwargs)
    dtaidistancec_dtw.dtw_print_wps_compact(&wps[0,0], l1, l2, &settings._settings)

def best_path_compact(seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2, **kwargs):
    cdef Py_ssize_t path_length;
    settings = DTWSettings(**kwargs)
    cdef Py_ssize_t *i1 = <Py_ssize_t *> PyMem_Malloc((l1 + l2) * sizeof(Py_ssize_t))
    if not i1:
        raise MemoryError()
    cdef Py_ssize_t *i2 = <Py_ssize_t *> PyMem_Malloc((l1 + l2) * sizeof(Py_ssize_t))
    if not i2:
        raise MemoryError()
    try:
        path_length = dtaidistancec_dtw.dtw_best_path(&wps[0,0], i1, i2, l1, l2,
                                                      &settings._settings)
        path = []
        for i in range(path_length):
            path.append((i1[i], i2[i]))
        path.reverse()
    finally:
        PyMem_Free(i1)
        PyMem_Free(i2)
    return path

def best_path_compact_affinity(seq_t[:, :] wps, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t rs, Py_ssize_t cs, **kwargs):
    cdef Py_ssize_t path_length;
    settings = DTWSettings(**kwargs)
    cdef Py_ssize_t *i1 = <Py_ssize_t *> PyMem_Malloc((l1 + l2) * sizeof(Py_ssize_t))
    if not i1:
        raise MemoryError()
    cdef Py_ssize_t *i2 = <Py_ssize_t *> PyMem_Malloc((l1 + l2) * sizeof(Py_ssize_t))
    if not i2:
        raise MemoryError()
    try:
        path_length = dtaidistancec_dtw.dtw_best_path_affinity(&wps[0,0], i1, i2, l1, l2,
                                                               rs, cs, &settings._settings)
        path = []
        for i in range(path_length):
            path.append((i1[i], i2[i]))
        path.reverse()
    finally:
        PyMem_Free(i1)
        PyMem_Free(i2)
    return path


def srand(unsigned int seed):
    dtaidistancec_dtw.dtw_srand(seed)

def warping_path_prob(seq_t[:] s1, seq_t[:] s2, seq_t avg, include_distance=False, **kwargs):
    # Assumes C contiguous
    cdef Py_ssize_t path_length;
    cdef seq_t dist;
    settings = DTWSettings(**kwargs)
    cdef Py_ssize_t *i1 = <Py_ssize_t *> PyMem_Malloc((len(s1) + len(s2)) * sizeof(Py_ssize_t))
    if not i1:
        raise MemoryError()
    cdef Py_ssize_t *i2 = <Py_ssize_t *> PyMem_Malloc((len(s1) + len(s2)) * sizeof(Py_ssize_t))
    if not i2:
        raise MemoryError()
    try:
        dist = dtaidistancec_dtw.dtw_warping_path_prob_ndim(&s1[0], len(s1), &s2[0], len(s2), i1, i2, &path_length,
                                                            avg, 1, &settings._settings)
        path = []
        for i in range(path_length):
            path.append((i1[i], i2[i]))
        path.reverse()
    finally:
        PyMem_Free(i1)
        PyMem_Free(i2)
    if include_distance:
        return path, dist
    return path

{% set suffix = '' %}
{%- include 'dtw_cc_distancematrix.jinja.pyx' %}

{% set suffix = '_ndim' %}
{%- include 'dtw_cc_distancematrix.jinja.pyx' %}


def distance_matrix_length(DTWBlock block, Py_ssize_t nb_series):
    cdef Py_ssize_t length
    length = dtaidistancec_dtw.dtw_distances_length(&block._block, nb_series, nb_series)
    return length

{% set suffix = '' %}
{%- include 'dtw_cc_dba.jinja.pyx' %}

{% set suffix = '_ndim' %}
{%- include 'dtw_cc_dba.jinja.pyx' %}


cdef dba_inner(cur, seq_t *c_ptr, Py_ssize_t c_len, unsigned char *mask_ptr, int nb_prob_samples, int ndim, DTWSettings settings):
    cdef seq_t *matrix_ptr;
    cdef DTWSeriesMatrix matrix
    cdef DTWSeriesMatrixNDim matrix_ndim
    cdef DTWSeriesPointers ptrs
    if isinstance(cur, DTWSeriesMatrix) or isinstance(cur, DTWSeriesPointers):
        pass
    elif cur.__class__.__name__ == "SeriesContainer":
        cur = cur.c_data_compat()
    else:
        cur = dtw_series_from_data(cur)

    if isinstance(cur, DTWSeriesPointers):
        ptrs = cur
        dtaidistancec_dtw.dtw_dba_ptrs(
            ptrs._ptrs, ptrs._nb_ptrs, ptrs._lengths,
            c_ptr, c_len, mask_ptr, nb_prob_samples, ndim, &settings._settings)
    elif isinstance(cur, DTWSeriesMatrix):
        matrix = cur
        matrix_ptr = &matrix._data[0, 0]
        dtaidistancec_dtw.dtw_dba_matrix(
            matrix_ptr, matrix.nb_rows, matrix.nb_cols,
            c_ptr, c_len, mask_ptr, nb_prob_samples, ndim, &settings._settings)
    elif isinstance(cur, DTWSeriesMatrixNDim):
        matrix_ndim = cur
        matrix_ptr = &matrix_ndim._data[0, 0, 0]
        dtaidistancec_dtw.dtw_dba_matrix(
            matrix_ptr, matrix_ndim.nb_rows, matrix_ndim.nb_cols,
            c_ptr, c_len, mask_ptr, nb_prob_samples, ndim, &settings._settings)
    else:
        raise ValueError("Series are not the expected type (DTWSeriesPointers, DTWSeriesMatrix "
                         "or DTWSeriesMatrixNDim): {}".format(type(cur)))
