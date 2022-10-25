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
      r is 0.0000001 if not given
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
            r = 0.0000001
        S = 1 / (r + D)
    elif method == 'reverse':
        if r is None:
            r = np.min(D) + np.max(D)
        S = r - D
    else:
        raise ValueError("method={} is not supported".format(method))
    return S
