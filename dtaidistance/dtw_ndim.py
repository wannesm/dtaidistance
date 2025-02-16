# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw_ndim
~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW) for N-dimensional series.

All the functionality in this subpackage is also available in
the dtw subpackage with argument ``use_ndim=True``.

:author: Wannes Meert
:copyright: Copyright 2017-2024 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import os
import logging

from . import dtw
from . import ed
from .dtw import SeriesContainer
from . import util_numpy
from . import innerdistance


logger = logging.getLogger("be.kuleuven.dtai.distance")
dtaidistance_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)


def ub_euclidean(s1, s2, inner_dist=innerdistance.default):
    """ Euclidean (dependent) distance between two n-dimensional sequences. Supports different lengths.

    If the two series differ in length, compare the last element of the shortest series
    to the remaining elements in the longer series.

    :param s1: Sequence of numbers, 1st dimension is sequence, 2nd dimension is n-dimensional value vector.
    :param s2: Sequence of numbers, 1st dimension is sequence, 2nd dimension is n-dimensional value vector.
    :return: Euclidean distance
    """
    return ed.distance(s1, s2, inner_dist=inner_dist, use_ndim=True)


def distance(s1, s2, window=None, max_dist=None,
             max_step=None, max_length_diff=None, penalty=None, psi=None,
             use_c=False, use_pruning=False, only_ub=False,
             inner_dist=innerdistance.default):
    """(Dependent) Dynamic Time Warping using multidimensional sequences.

    Assumes the first dimension to be the sequence item index, and the second
    dimension is the index of the value in the vector.

    Example:

    ::

        s1 = np.array([[0, 0],
                       [0, 1],
                       [2, 1],
                       [0, 1],
                       [0, 0]], dtype=np.double)
        s2 = np.array([[0, 0],
                       [2, 1],
                       [0, 1],
                       [0, .5],
                       [0, 0]], dtype=np.double)
        d = distance(s1, s2)

    See :py:meth:`dtaidistance.dtw.distance` for parameters.

    This method returns the dependent DTW (DTW_D) [1] distance between two
    n-dimensional sequences. If you want to compute the independent DTW
    (DTW_I) distance, use the 1-dimensional version:

    ::

        dtw_i = 0
        for dim in range(ndim):
            dtw_i += dtw.distance(s1[:,dim], s2[:,dim])

    Note:
    If you are using the C-optimized code, the above snippet will trigger a
    copy operation to guarantee the arrays to be C-ordered and will thus create
    time and memory overhead. This can be avoided
    by storing the dimensions as separate arrays or by flipping the array dimensions
    and use dtw.distance(s1[dim,:], dtw.distance(s2[dim,:]).

    [1] M. Shokoohi-Yekta, B. Hu, H. Jin, J. Wang, and E. Keogh.
    Generalizing dtw to the multi-dimensional case requires an adaptive approach.
    Data Mining and Knowledge Discovery, 31:1–31, 2016.
    """
    return dtw.distance(s1, s2, window=window, max_dist=max_dist,
             max_step=max_step, max_length_diff=max_length_diff, penalty=penalty, psi=psi,
             use_c=use_c, use_pruning=use_pruning, only_ub=only_ub,
             inner_dist=inner_dist, use_ndim=True)


def distance_fast(s1, s2, window=None, max_dist=None,
                  max_step=None, max_length_diff=None, penalty=None, psi=None, use_pruning=False, only_ub=False,
                  inner_dist=innerdistance.default):
    """Fast C version of :meth:`distance`.

    Note: the series are expected to be arrays of the type ``double``.
    Thus ``numpy.array([[1,1],[2,2],[3,3]], dtype=numpy.double)``    """
    return dtw.distance_fast(s1, s2, window=window, max_dist=max_dist,
                             max_step=max_step, max_length_diff=max_length_diff, penalty=penalty,
                             psi=psi, use_pruning=use_pruning, only_ub=only_ub,
                             inner_dist=inner_dist, use_ndim=True)


def warping_paths(*args, **kwargs):
    """
    Dynamic Time Warping (keep full matrix) using multidimensional sequences.

    See :py:meth:`dtaidistance.dtw.warping_paths` for parameters.
    """
    return dtw.warping_paths(*args, use_ndim=True, **kwargs)


def warping_paths_fast(*args, **kwargs):
    """
    Dynamic Time Warping (keep full matrix) using multidimensional sequences.

    See :py:meth:`dtaidistance.dtw.warping_paths` for parameters.
    """
    return dtw.warping_paths_fast(*args, use_ndim=True, **kwargs)


def distance_matrix(s, ndim=None, max_dist=None, use_pruning=False, max_length_diff=None,
                    window=None, max_step=None, penalty=None, psi=None,
                    block=None, compact=False, parallel=False,
                    use_c=False, use_mp=False, show_progress=False, only_triu=False,
                    inner_dist=innerdistance.default):
    """Distance matrix for all n-dimensional sequences in s.

    This method returns the dependent DTW (DTW_D) [1] distance between two
    n-dimensional sequences. If you want to compute the independent DTW
    (DTW_I) distance, use the 1-dimensional version and sum the distance matrices:

    ::

        dtw_i = dtw.distance_matrix(series_sep_dim[0])
        for dim in range(1, ndim):
            dtw_i += dtw.distance_matrix(series_sep_dim(dim)

    Where series_sep_dim is a datastructure that returns a list of the sequences that
    represents the i-th dimension of each sequence in s.

    :param s: Iterable of series
    :param window: see :meth:`distance`
    :param max_dist: see :meth:`distance`
    :param max_step: see :meth:`distance`
    :param max_length_diff: see :meth:`distance`
    :param penalty: see :meth:`distance`
    :param psi: see :meth:`distance`
    :param block: Only compute block in matrix. Expects tuple with begin and end, e.g. ((0,10),(20,25)) will
        only compare rows 0:10 with rows 20:25.
    :param compact: Return the distance matrix as an array representing the upper triangular matrix.
    :param parallel: Use parallel operations
    :param use_c: Use c compiled Python functions
    :param use_mp: Use Multiprocessing for parallel operations (not OpenMP)
    :param show_progress: Show progress using the tqdm library. This is only supported for
        the pure Python version (thus not the C-based implementations).
    :param only_triu: Only fill the upper triangle
    :returns: The distance matrix or the condensed distance matrix if the compact argument is true

    [1] M. Shokoohi-Yekta, B. Hu, H. Jin, J. Wang, and E. Keogh.
    Generalizing dtw to the multi-dimensional case requires an adaptive approach.
    Data Mining and Knowledge Discovery, 31:1–31, 2016.
    """
    # Check whether multiprocessing is available
    s = SeriesContainer.wrap(s)
    s.set_detected_ndim(ndim)
    return dtw.distance_matrix(s, max_dist=max_dist, use_pruning=use_pruning, max_length_diff=max_length_diff,
                    window=window, max_step=max_step, penalty=penalty, psi=psi,
                    block=block, compact=compact, parallel=parallel,
                    use_c=use_c, use_mp=use_mp, show_progress=show_progress, only_triu=only_triu,
                    inner_dist=inner_dist, use_ndim=True)


def distance_matrix_fast(s, ndim=None, max_dist=None, max_length_diff=None,
                         window=None, max_step=None, penalty=None, psi=None,
                         block=None, compact=False, parallel=True, only_triu=False,
                         inner_dist=innerdistance.default, use_c=True):
    """Fast C version of :meth:`distance_matrix`."""
    return distance_matrix(s, ndim=ndim, max_dist=max_dist, max_length_diff=max_length_diff,
                           window=window, max_step=max_step, penalty=penalty, psi=psi,
                           block=block, compact=compact, parallel=parallel,
                           use_c=True, show_progress=False, only_triu=only_triu,
                           inner_dist=inner_dist)


def warping_path(from_s, to_s, **kwargs):
    """Compute warping path between two sequences."""
    return dtw.warping_path(from_s, to_s, use_ndim=True, **kwargs)
