"""
dtaidistance.dtw_cc_numpy
~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW), C implementation, numpy specific extensions.

:author: Wannes Meert
:copyright: Copyright 2020-2022 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import dtw_cc
from libc.stdint cimport intptr_t
import numpy as np
cimport numpy as np
import logging


logger = logging.getLogger("be.kuleuven.dtai.distance")


DTYPE = np.double
ctypedef np.double_t DTYPE_t


def dtw_series_from_numpy(np.ndarray[DTYPE_t, ndim=2] data):
    if not data.flags.c_contiguous:
            logger.debug("Warning: The numpy array or matrix passed to method distance_matrix is not C-contiguous. " +
                         "The array will be copied.")
            data = data.copy(order='C')
    ptrs = dtw_cc.DTWSeriesMatrix(data)
    return ptrs


def dtw_series_from_numpy_ndim(np.ndarray[DTYPE_t, ndim=3] data):
    if not data.flags.c_contiguous:
            logger.debug("Warning: The numpy array or matrix passed to method distance_matrix is not C-contiguous. " +
                         "The array will be copied.")
            data = data.copy(order='C')
    ptrs = dtw_cc.DTWSeriesMatrixNDim(data)
    return ptrs
