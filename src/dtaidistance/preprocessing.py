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
    axis = 0
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

    Since version 2.4 the filter uses Gustafsson’s method for handling the edges
    of the series.

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
                         "Requires 0<smooth<0.5 (now Smooth={}, Wn={})".format(smooth, Wn)) from exc
    try:
        series = signal.filtfilt(b, a, series, axis=axis, method="gust")
    except ValueError as exc:
        raise ValueError("Cannot apply smoothing, "
                         "see the Scipy exception above to solve the problem "
                         "or disable smoothing by setting smooth to None") from exc
    return series


def derivative(series, smooth=None):
    """Derivative series.

    dq = ((q_i - q_{i-1}) + (q_{i+1} - q_{i-1})/2)/2

    Based on Keogh, E. and Pazzani, M. "Derivative Dynamic Time Warping".
    SIAM International Conference on Data Mining, 2002.

    The smoothing argument is used to smooth after computing the derivative. To apply the
    smoothing as explained in Keogh et al. (2002) one should do exponential smoothing before
    applying this method.

    :param series: Time series (must be numpy compatible)
    :param smooth: Smooth the derivative series by removing the highest frequencies.
        The cut-off frequency is computed using the `smooth` argument. This
        fraction (a number between 0.0 and 0.5) of the highest frequencies is
        removed.
    :return: Differenced Numpy array of length len(series) - 1
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Differencing requires Numpy")
    axis = 0
    if isinstance(series, np.ndarray):
        if len(series.shape) == 1:
            axis = 0
        else:
            axis = 1

    if axis == 0:
        qim = series[:-2]
        qi = series[1:-1]
        qip = series[2:]
    else:
        raise NotImplementedError("Derivative for axis!=0 is not yet implemented")

    seriesd = np.zeros(series.shape[axis])
    seriesd[1:-1] = np.add(np.subtract(qi, qim), (np.subtract(qip, qim) / 2)) / 2
    if axis == 0:
        seriesd[0] = series[1] - series[0]
        seriesd[-1] = series[-1] - series[-2]
    else:
        raise NotImplementedError("Derivative for axis!=0 is not yet implemented")

    if smooth is not None:
        seriesd = smoothing(seriesd, smooth)
    return seriesd


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


def mixedlinearlogdomain(series, c=10):
    """Transform to a mixture between linear and log domain
    (and retain the sign of the signal).

    The transformation is (for positive values):
    x            if x<=c
    c+ln(x-c+1)  if x>c

    :type c: Union[int,list]
    :param series: Time series (must be numpy compatible)
    :param c: Switch between linear to log domain at this value, should be <= 1.
        If two numbers are given as a tuple, the first one is used for positive
        values, the second for negative values.
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Transforming to log domain requires Numpy")

    if type(c) in [tuple, list]:
        pos = np.heaviside(series, 1)
        seriesp = pos*series
        seriesn = (1-pos)*np.abs(series)
        cc = c[0]
        step = np.heaviside(seriesp - cc, 1)
        seriesp = (1 - step) * seriesp + step * (cc + np.log1p(step * (seriesp - cc)))
        cc = -c[1]
        step = np.heaviside(seriesn - cc, 1)
        seriesn = (1 - step) * seriesn + step * (cc + np.log1p(step * (seriesn - cc)))
        series = -seriesn + seriesp
    else:
        sign = np.sign(series)
        series = np.abs(series)
        step = np.heaviside(series-c, 1)
        # should be vectorized
        # step is in log1p to avoid nan
        series = sign * ((1-step)*series + step*(c+np.log1p(step*(series-c))))
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
