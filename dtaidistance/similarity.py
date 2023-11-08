try:
    import numpy as np
except ImportError:
    np = None


def distance_to_similarity(D, r=None, method='exponential', return_params=False):
    """Transform a distance matrix to a similarity matrix.

    The available methods are:
    - Exponential: e^(-D / r)
      r is max(D) if not given
    - Gaussian: e^(-D^2 / r^2)
      r is max(D) if not given
    - Reciprocal: 1 / (r + D)
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
    :return: Similarity matrix S
    """
    method = method.lower()
    if method == 'exponential':
        if r is None:
            r = np.max(D)
        S = np.exp(-D / r)
    elif method == 'gaussian':
        if r is None:
            r = np.max(D)
        S = np.exp(-np.power(D, 2) / r**2)
    elif method == 'reciprocal':
        if r is None:
            r = 1
        S = 1 / (r + D)
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


def squash(X, r=None, base=None, x0=None, method="logistic", return_params=False):
    """Squash a function monotonically to a range between 0 and 1.

    The available methods are:
    - Logistic: 1 / (1 + e^(-(X-x0) / r)
    - Gaussian: e^(-(X-x0)^2 / r^2)

    Example usage::

        dist_matrix = dtw.distance_matrix(series)
        dist_matrix_sq = squash(dist_matrix)


    Based on:
    Vercruyssen, V., Meert, W., Verbruggen, G., Maes, K., Baumer, R., & Davis, J.
    (2018). Semi-supervised anomaly detection with an application to water analytics.
    In 2018 IEEE international conference on data mining (ICDM) (Vol. 2018, pp. 527-536)

    :param X: Distances values
    :param r: The slope of the squashing (see the formula above)
    :param x0: The midpoint of the squashing (see the formula above)
    :param method: The choice of sqaush function: logistic or gaussian
    :param return_params: Also return the used values for r and X0
    """
    result = None
    if method == "gaussian":
        x0 = 0  # not supported for gaussian
        if r is None:
            r = 1
        if base is None:
            result = 1 - np.exp(-np.power(X - x0, 2) / r**2)
        else:
            result = 1 - np.power(base, -np.power(X - x0, 2) / r**2)
    elif method == "logistic":
        if x0 is None:
            x0 = np.mean(X)
        if r is None:
            r = x0 / 6
        if base is None:
            result = 1 / (1 + np.exp(-(X - x0) / r))
        else:
            result = 1 / (1 + np.power(base, -(X - x0) / r))
    else:
        raise ValueError("Unknown value for method")
    if return_params:
        return result, r, x0
    else:
        return result
