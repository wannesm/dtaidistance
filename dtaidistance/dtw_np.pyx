"""
dtaidistance.dtw_cc
~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW), Numpy entries.

:author: Wannes Meert
:copyright: Copyright 2017-2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging

import numpy as np
cimport numpy as np

cimport dtw_cc


logger = logging.getLogger("be.kuleuven.dtai.distance")


DTYPE = np.double
ctypedef np.double_t DTYPE_t
cdef double inf = np.inf


def distance(np.ndarray[DTYPE_t, ndim=1] s1, np.ndarray[DTYPE_t, ndim=1] s2, **kwargs):
    """DTW distance.

    See distance(). This calls a pure c dtw computation that avoids the GIL.

    Performs some Numpy specific checks before passing the data to the C library.

    :param s1: First sequence (buffer of doubles)
    :param s2: Second sequence (buffer of doubles)
    """
    # If the arrays (memoryviews) are not C contiguous, the pointer will not point to the correct array
    if not s1.base.flags.c_contiguous:
        logger.debug("Warning: Sequence 1 passed to method distance is not C-contiguous. " +
                     "The sequence will be copied.")
        s1 = s1.copy()
    if isinstance(s2, (np.ndarray, np.generic)):
        if not s2.base.flags.c_contiguous:
            logger.debug("Warning: Sequence 2 passed to method distance is not C-contiguous. " +
                         "The sequence will be copied.")
            s2 = s2.copy()
    dtw_cc.distance(s1, s2, **kwargs)
