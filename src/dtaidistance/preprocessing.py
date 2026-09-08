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


def _least_squares(x, y):
    """a*x + b

    :returns: [b, a]
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Z normalization requires Numpy")
    if np is None:
        raise NumpyException("Least squares requires Numpy")
    if len(x.shape) > 1 and x.shape[1] == 1:
        x = x.reshape(-1)
    X_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((np.multiply(x - X_mean, y - y_mean)))
    denominator = np.sum(np.power(x - X_mean, 2))
    a = numerator / denominator
    b = y_mean - a * X_mean
    return [b, a]


def _total_least_squares(X,y):
    """
    Total Least Squares.

    Use SVD to solve min ||[X|y] - [X'|y']||_F subject to y' = X'b.

    :param X: array-like, shape (n_samples, n_features)
        Input data matrix.
    :param y: array-like, shape (n_samples,)
        Target vector.
    :returns: [b, a, ...] where y= b + a*x + ...
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Z normalization requires Numpy")
    if np is None:
        raise NumpyException("Total least squares requires Numpy")
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    # we also want the intercept, add a column of ones
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    Z = np.hstack((X, y))
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt.T
    v = V[:, -1]
    v_x = v[:-1]
    v_y = v[-1]
    b_tls = -v_x / v_y
    return b_tls


def scale_linearly(src_ts, tgt_ts, quantile=0.05, nb_steps=10,
                   use_totalleastsquares=True, use_znormalization=True):
    """Scale the amplitude of src_ts timeseries linearly to match the
    tgt_ts timeries. An affine transformation is applied to the src_ts.
    
    If the time series are multivariate, the scaling is applied independently
    to each variate.

    :param src_ts: array-like, shape (n_samples, n_variates) or (n_samples,)
        Source time series.
    :param tgt_ts: array-like, shape (n_samples, n_variates) or (n_samples,)
        Target time series.
    :param quantile: Which quantile of the points in the time series
        should be used.
    :param nb_steps: How many steps to use to reduce the number of points
        in the time series to reach the amount specified by the quantile.
    :param use_totalleastsquares:
    :param use_znormalization: First perform z-normalization. If the values
        of one series are larger, this gets more weight in the least squares.
    :returns: A scaled src_ts
    """
    try:
        import numpy as np
    except ImportError:
        raise NumpyException("Z normalization requires Numpy")
    if np is None:
        raise NumpyException("Total least squares requires Numpy")
    if len(src_ts.shape) == 1:
        return_1d = True
        src_ts = src_ts.reshape(-1,1)
    else:
        return_1d = False
    if use_totalleastsquares:
        ls = _total_least_squares
    else:
        ls = _least_squares
    if len(src_ts.shape) == 1:
        src_ts2 = src_ts.reshape(-1,1).copy()
    else:
        src_ts2 = src_ts.copy()
    if len(tgt_ts.shape) == 1:
        tgt_ts2 = tgt_ts.reshape(-1,1)
    else:
        tgt_ts2 = tgt_ts
    nb_inst_rm = int(len(src_ts2)*(1-quantile)/(nb_steps-1))
    if src_ts2.shape[1] != tgt_ts2.shape[1]:
        raise ValueError(
            f"Source and target time series have different "
            f"dimensions: {src_ts2.shape} != {tgt_ts2.shape}")
    for i_var in range(src_ts2.shape[1]):
        nb_inst = len(src_ts2)
        src_ts3 = src_ts2[:,i_var].reshape(-1,1)
        tgt_ts3 = tgt_ts2[:,i_var]
        if use_znormalization:
            src_mean, src_std = np.mean(src_ts3), np.std(src_ts3)
            tgt_mean, tgt_std = np.mean(tgt_ts3), np.std(tgt_ts3)
            src_ts3 = (src_ts3 - src_mean) / src_std
            tgt_ts3 = (tgt_ts3 - tgt_mean) / tgt_std
        else:
            src_mean, src_std = None, None
            tgt_mean, tgt_std = None, None
        src_ts4 = src_ts3
        tgt_ts4 = tgt_ts3
        b, a = 0, 1
        dss = np.sum(np.abs(tgt_ts4 - src_ts4))
        for i_step in range(nb_steps):
            b2, a2 = ls(src_ts4, tgt_ts4)
            if a2 < -0.01:
                # If the signal switches, stop
                break
            yp = a2*src_ts4.reshape(-1)+b2
            ds = np.abs(tgt_ts4 - yp)
            dss2 = np.sum(ds)
            if dss2 > 1.001*dss:
                # If dist goes up, stop
                break
            a, b, dss = a2, b2, dss2
            if i_step == nb_steps - 1:
                break
            nb_inst -= nb_inst_rm
            idx2 = np.argpartition(ds, nb_inst)[:nb_inst]
            src_ts4 = src_ts4[idx2]
            tgt_ts4 = tgt_ts4[idx2]
        if use_znormalization:
            a = a*tgt_std/src_std
            b = tgt_std*b + tgt_mean - a*src_mean
        src_ts2[:,i_var] = a * src_ts2[:,i_var] + b
    if return_1d:
        src_ts2 = src_ts2.reshape(-1)
    return src_ts2

