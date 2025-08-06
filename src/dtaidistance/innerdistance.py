# -*- coding: UTF-8 -*-
"""
dtaidistance.innerdistance
~~~~~~~~~~~~~~~~~~~~~~~~~~

Inner (or local) distances for DTW and ED

We make the distinction between three operations:

- The `inner_dist`: The inner distance (or local distance, or cost) is
    the function that expresses the similarity between two values in the
    series in the cost space.
- The `result`: The transformation to change from the cost space to
    the (global) distance space.
- The `inner_val`: The transformation to change from the (global) distance
    space to the cost space.

:author: Wannes Meert
:copyright: Copyright 2023-2025 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import math
import logging

from . import util
from . import util_numpy

# The available inner distances, with their integer identifier are:
# - 0: squared Euclidean (for univariate series this is squared difference)
# - 1: Euclidean (for univariate series this his absolute difference)

try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
    DTYPE = np.double
    argmin = np.argmin
    argmax = np.argmax
    array_min = np.min
    array_max = np.max
except ImportError:
    np = None
    argmin = util.argmin
    argmax = util.argmax
    array_min = min
    array_max = max


logger = logging.getLogger("be.kuleuven.dtai.distance")
default = 'squared euclidean'


class SquaredEuclidean:
    """Squared Euclidean inner distance for univariate series.
    This is identical to squared difference.
    """

    @staticmethod
    def inner_dist(x, y):
        return (x - y) ** 2

    @staticmethod
    def result(x):
        if np is not None and isinstance(x, np.ndarray):
            return np.sqrt(x)
        return math.sqrt(x)

    @staticmethod
    def inner_val(x):
        return x*x


class SquaredEuclideanNdim:
    """Squared Euclidean inner distance for multivariate series."""

    @staticmethod
    def inner_dist(x, y):
        return np.sum((x - y) ** 2)

    @staticmethod
    def result(x):
        return np.sqrt(x)

    @staticmethod
    def inner_val(x):
        return x * x


class Euclidean:
    """Euclidean inner distance for univariate series.
    This is identical to absoluate difference.
    """

    @staticmethod
    def inner_dist(x, y):
        return abs(x - y)

    @staticmethod
    def result(x):
        return x

    @staticmethod
    def inner_val(x):
        return x


class EuclideanNdim:
    """Squared Euclidean inner distance for multivariate series."""

    @staticmethod
    def inner_dist(x, y):
        return np.sqrt(np.sum(np.power(x - y, 2)))

    @staticmethod
    def result(x):
        return x

    @staticmethod
    def inner_val(x):
        return x


class CustomInnerDist:
    """API to create your own custom inner (local)distance."""

    @staticmethod
    def inner_dist(x, y):
        """The cost (or inner distance, or local distance) between two points in the series.

        For n-dimensional data, the two arguments x and y will be vectors.
        Otherwise, they are scalars.

        For example, for default DTW this would be the Squared Euclidean
        distance: (a-b)**2.
        """
        raise Exception("Function not defined")

    @staticmethod
    def result(x):
        """The transformation applied to the sum of all inner distances.
        We also refer to this method as cost-to-dist.
        The inverse operation of `inner_val`.

        The variable x can be both a single number as a matrix.

        For example, for default DTW, which uses Squared Euclidean, this
        would be: sqrt(d). Because d = (a_0-b_0)**2 + (a_1-b_1)**2 ...
        """
        raise Exception("Function not defined")

    @staticmethod
    def inner_val(x):
        """The transformation applied to input settings like max_step.
        We also refer to this method as dist-to-cost.
        The inverse operation of `result`."""
        raise Exception("Function not defined")


def inner_dist_cls(inner_dist="squared euclidean", use_ndim=False):
    if inner_dist == "squared euclidean" or inner_dist == 0:
        if use_ndim:
            use_cls = SquaredEuclideanNdim
        else:
            use_cls = SquaredEuclidean
    elif inner_dist == "euclidean" or inner_dist == 1:
        if use_ndim:
            use_cls = EuclideanNdim
        else:
            use_cls = Euclidean
    elif hasattr(inner_dist, 'inner_dist') and hasattr(inner_dist, 'result'):
        use_cls = inner_dist
    else:
        raise AttributeError(f"Unknown value for argument inner_dist: {inner_dist}")
    return use_cls


def inner_dist_fns(inner_dist="squared euclidean", use_ndim=False):
    """Return the inner distance (or local distance) functions and transformations.

    :param inner_dist: Type of inner_dist
    :param use_ndim: Use multivariate or not
    :return: (inner_dist, cost2dist, dist2cost)
    """
    use_cls = inner_dist_cls(inner_dist, use_ndim)
    return use_cls.inner_dist, use_cls.result, use_cls.inner_val


def to_c(inner_dist):
    if inner_dist == 'squared euclidean' or inner_dist == 0:
        return 0
    elif inner_dist == 'euclidean' or inner_dist == 1:
        return 1
    elif hasattr(inner_dist, 'inner_dist') and hasattr(inner_dist, 'result'):
        raise AttributeError('Custom inner distance functions are not supported for the fast C implementation')
    else:
        raise AttributeError('Unknown inner_dist: {}'.format(inner_dist))
