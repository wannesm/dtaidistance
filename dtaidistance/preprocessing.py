# -*- coding: UTF-8 -*-
"""
dtaidistance.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Preprocessing time series.

:author: Wannes Meert
:copyright: Copyright 2021-2024 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
from .exceptions import NumpyException, ScipyException


def differencing(series, smooth=None, diff_args=None):
    """Differencing series.

    :param series: Time series (must be numpy compatible)
    :param smooth: Smooth the differenced series by removing the highest frequencies.
        The cut-off frequency is computed using the `smooth` argument. This
        fraction (a number between 0.0 and 0.5) of the highest frequencies is
        removed.
    :param diff_args: Arguments to pass the numpy.diff
    :return: Differenced Numpy array of length len(series) - 1
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Differencing requires Numpy")
    if isinstance(series, np.ndarray):
        if len(series.shape) == 1:
            axis = 0
        else:
            axis = 1
    if diff_args is None:
        diff_args = {}
    series = np.diff(series, n=1, axis=axis, **diff_args)
    if smooth is not None:
        series = smoothing(series, smooth)
    return series


def smoothing(series, smooth):
    """Smooth the series.

    :param series: Time series (must be numpy compatible)
    :param smooth: Smooth the series by removing the highest frequencies.
        The cut-off frequency is computed using the `smooth` argument. This
        fraction (a number between 0.0 and 0.5) of the highest frequencies is
        removed.
    :return: Smoothed series as Numpy array
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Smoothing requires Numpy")
    try:
        from scipy import signal
    except ImportError:
        raise ScipyException("Smoothing requires Scipy")
    if isinstance(series, np.ndarray):
        if len(series.shape) == 1:
            axis = 0
        else:
            axis = 1
    else:
        axis = 0
    fs = 100  # sample rate, Hz
    cutoff = fs * smooth  # cut-off frequency, Hz
    nyq = 0.5 * fs  # Nyquist frequency
    Wn = cutoff / nyq
    try:
        b, a = signal.butter(N=2, Wn=Wn, btype='low', analog=False, output='ba')
    except ValueError as exc:
        raise ValueError("Cannot construct filter, change the smoothing factor. "
                         f"Requires 0<smooth<0.5 (now {smooth=}, {Wn=})") from exc
    try:
        series = signal.filtfilt(b, a, series, axis=axis, method="gust")
    except ValueError as exc:
        raise ValueError("Cannot apply smoothing, "
                         "see the Scipy exception above to solve the problem "
                         "or disable smoothing by setting smooth to None") from exc
    return series


def logdomain(series):
    """Transform to the log domain and retain the sign of the signal.

    :param series: Time series (must be numpy compatible)
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Transforming to log domain requires Numpy")
    series = np.sign(series) * np.log1p(np.abs(series))
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
