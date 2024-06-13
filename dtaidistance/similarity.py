try:
    import numpy as np
except ImportError:
    np = None


def distance_to_similarity(D, r=None, a=None, method='exponential', return_params=False, cover_quantile=False):
    """Transform a distance matrix to a similarity matrix.

    The available methods are:
    - Exponential: e^(-D / r)
      r is max(D) if not given
    - Gaussian: e^(-D^2 / r^2)
      r is max(D) if not given
    - Reciprocal: 1 / (r + D*a)
      r is 1 if not given
    - Reverse: r - D
      r is min(D) + max(D) if not given

    All of these methods are monotonically decreasing transformations. The order of the
    distances thus remains unchanged (only the direction).

    Example usage::

        dist_matrix = dtw.distance_matrix(series)
        sim_matrix = distance_to_similarity(dist_matrix)


    :param D: The distance matrix
    :param r: A scaling or smoothing parameter.
    :param method: One of 'exponential', 'gaussian', 'reciprocal', 'reverse'
    :param return_params: Return the value used for parameter r
    :param cover_quantile: Compute r such that the function covers the `cover_quantile` fraction of the data.
        Expects a value in [0,1]. If a tuple (quantile, value) is given, then r (and a) are set such that
        at the quantile the given value is reached (if not given, value is 1-quantile).
    :return: Similarity matrix S
    """
    if cover_quantile is not False:
        if type(cover_quantile) in [tuple, list]:
            cover_quantile, cover_quantile_target = cover_quantile
        else:
            cover_quantile_target = 1 - cover_quantile
    else:
        cover_quantile_target = None
    method = method.lower()
    if method == 'exponential':
        if r is None:
            if cover_quantile is False:
                r = np.max(D)
            else:
                r = -np.quantile(D, cover_quantile) / np.log(cover_quantile_target)
        S = np.exp(-D / r)
    elif method == 'gaussian':
        if r is None:
            if cover_quantile is False:
                r = np.max(D)
            else:
                r = np.sqrt(-np.quantile(D, cover_quantile) ** 2 / np.log(cover_quantile_target))
        S = np.exp(-np.power(D, 2) / r**2)
    elif method == 'reciprocal':
        if r is None:
            r = 1
        if a is None:
            if cover_quantile is False:
                a = 1
            else:
                a = (1 - cover_quantile_target * r) / (cover_quantile_target * np.quantile(D, cover_quantile))
        S = 1 / (r + D*a)
    elif method == 'reverse':
        if r is None:
            r = np.min(D) + np.max(D)
        S = (r - D) / r
    else:
        raise ValueError("method={} is not supported".format(method))
    if return_params:
        return S, r
    else:
        return S


def squash(X, r=None, base=None, x0=None, method="logistic", return_params=False, keep_sign=False,
           cover_quantile=False):
    """Squash a function monotonically to a range between 0 and 1.

    The available methods are:
    - Logistic: 1 / (1 + e^(-(X-x0) / r)
    - Gaussian: 1 - e^(-(X-x0)^2 / r^2)
    - Exponential: 1 - e^(-(X-x0) / r)

    Example usage::

        dist_matrix = dtw.distance_matrix(series)
        dist_matrix_sq = squash(dist_matrix)


    Based on:
    Vercruyssen, V., Meert, W., Verbruggen, G., Maes, K., Baumer, R., & Davis, J.
    (2018). Semi-supervised anomaly detection with an application to water analytics.
    In 2018 IEEE international conference on data mining (ICDM) (Vol. 2018, pp. 527-536)

    :param X: Distances values
    :param r: The slope of the squashing (see the formula above)
    :param base: Use this value as base instead of e
    :param x0: The midpoint of the squashing (see the formula above)
    :param method: The choice of sqaush function: logistic, gaussian, or exponential
    :param keep_sign: Negative values should stay negative
    :param return_params: Also return the used values for r and X0
    :param cover_quantile: Compute r such that the function covers the `cover_quantile` fraction of the data.
        Expects a value in [0,1]. If a tuple (quantile, value) is given, then r (and a) are set such that
        at the quantile the given value is reached (if not given, value is quantile).
    """
    if cover_quantile is not False:
        if type(cover_quantile) in [tuple, list]:
            cover_quantile, cover_quantile_target = cover_quantile
        else:
            cover_quantile_target = cover_quantile
    else:
        cover_quantile_target = None
    result = None
    if keep_sign:
        Xs = np.sign(X)
        Xz = 0
        X = np.abs(X)
    else:
        Xs = 1
    if method == "gaussian":
        x0 = 0  # not supported for gaussian
        if r is None:
            if cover_quantile is False:
                r = 1
            else:
                r = np.sqrt(-(np.quantile(X, cover_quantile)-x0)**2/np.log(1-cover_quantile_target))
        if base is None:
            result = 1 - np.exp(-np.power(X - x0, 2) / r**2)
            Xz = 1 - np.exp(-np.power(0 - x0, 2) / r**2)
        else:
            result = 1 - np.power(base, -np.power(X - x0, 2) / r**2)
            Xz = 1 - np.power(base, -np.power(0 - x0, 2) / r**2)
    if method == "exponential":
        x0 = 0  # not supported for exponential
        if r is None:
            if cover_quantile is False:
                r = 1
            else:
                r = -(np.quantile(X, cover_quantile)-x0)/np.log(1-cover_quantile_target)
        if base is None:
            result = 1 - np.exp(-(X - x0) / r)
            Xz = 1 - np.exp(x0 / r)
        else:
            result = 1 - np.power(base, -(X - x0) / r)
            Xz = 1 - np.power(base, x0 / r)
    elif method == "logistic":
        if x0 is None:
            x0 = np.mean(X)
        if r is None:
            if cover_quantile is False:
                r = x0 / 6
            else:
                r = -(np.quantile(X, cover_quantile)-x0) / np.log(1/cover_quantile_target-1)
        if base is None:
            result = 1 / (1 + np.exp(-(X - x0) / r))
            Xz = 1 / (1 + np.exp(-(0 - x0) / r))
        else:
            result = 1 / (1 + np.power(base, -(X - x0) / r))
            Xz = 1 / (1 + np.power(base, -(0 - x0) / r))
    else:
        raise ValueError("Unknown value for method")
    if keep_sign:
        result = Xs * (result - Xz)
    if return_params:
        return result, r, x0
    else:
        return result
