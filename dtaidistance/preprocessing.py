# -*- coding: UTF-8 -*-
"""
dtaidistance.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Preprocessing time series.

:author: Wannes Meert
:copyright: Copyright 2021 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
from .exceptions import NumpyException, ScipyException


def differencing(series, smooth=None):
    """Differencing series.

    :param series: Time series (must be numpy compatible)
    :param smooth: Smooth the series by removing the `smooth` percentage ([0-1])
        of highest frequency.
    :return: Differenced Numpy array
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Differencing requires Numpy")
    try:
        from scipy import signal
    except ImportError:
        raise ScipyException("Differencing requires Scipy")
    series = np.diff(series, n=1, axis=1)
    if smooth is not None:
        fs = 100  # sample rate, Hz
        cutoff = fs * smooth  # cut off frequency, Hz
        nyq = 0.5 * fs  # Nyquist frequency
        b, a = signal.butter(2, cutoff / nyq, btype='low', analog=False, output='ba')
        series = signal.filtfilt(b, a, series, axis=1)
    return series


def znormal(series):
    """Z-normalize the time series.

    :param series: Time series (must be a numpy compatible)
    :return: Z-normalized Numpy array
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Z normalization requires Numpy")
    series = np.array(series)
    series = (series - series.mean(axis=1)[:, None]) / series.std(axis=1)[:, None]
    return series
