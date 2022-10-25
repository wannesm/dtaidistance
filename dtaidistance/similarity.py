try:
    import numpy as np
except ImportError:
    np = None


def distance_to_similarity(D, r=None, method='exponential'):
    """Transform a distance matrix to a similarity matrix.

    The avaiable methods are:
    - Exponential: e^(-D / r)
      r is max(D) if not given
    - Reciprocal: 1 / (r + D)
      r is 1 if not given
    - Reverse: r - D
      r is min(D) + max(D) if not given

    :param D: The distance matrix
    :param r: A scaling or smoothing parameter.
    :param method: One of 'exponential', 'reciprocal', 'reverse'
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
        S = np.exp(-np.power(D, 2) / r)
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
    return S


def squash(X, r=None, base=None, x0=0, method="logistic"):
    """Squash a function monotonically to a range between 0 and 1.

    Based on:
    Vercruyssen, V., Meert, W., Verbruggen, G., Maes, K., Baumer, R., & Davis, J.
    (2018). Semi-supervised anomaly detection with an application to water analytics.
    In 2018 IEEE international conference on data mining (ICDM) (Vol. 2018, pp. 527-536)
    """
    if method == "gaussian":
        x0 = 0  # not supported for gaussian
        if r is None:
            r = 1
        if base is None:
            return 1 - np.exp(-np.power(X - x0, 2) / r**2)
        return 1 - np.power(base, -np.power(X - x0, 2) / r**2)
    elif method == "logistic":
        if r is None:
            r = 1
        if base is None:
            return 1 / (1 + np.exp(-(X - x0) / r))
        return 1 / (1 + np.power(base, -(X - x0) / r))
