# -*- coding: UTF-8 -*-
"""
dtaidistance.postprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Postprocessing similarity measures (e.g. paths).

:author: Wannes Meert
:copyright: Copyright 2025 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
from .innerdistance import inner_dist_fns
from .preprocessing import derivative


def distance_from_path(ts1, ts2, path, inner_dist="squared euclidean",
                       relaxed=True):
    """Compute the distance defined by a given warping path.

    If relaxation is true, the difference between two points that are
    warped to each other is computed more relaxed to take into account
    the shape of the time series. If a point appears in a time series
    where the series is steep (i.e., a high derivative), then where the
    timepoint was sampled has a larger impact than for a flat part. To
    compensate for this, we assume that the time point could have been
    anywhere in the time range [t-Delta_t/2,t+Delta_t/2], where Delta_t
    is the timestep (since we do not take explicit timepoints, here
    Delta_t = 1). The smallest difference between all possible values
    in this time range is used.

    :param ts1: The first time series
    :param ts2: The first time series
    :param path: A list of points (index_ts1, index_ts2)
    :param inner_dist: One of :class:`InnerDistType`
    :param relaxed: Take the shape of the time series into account
    """
    inner_dist, result, _ = inner_dist_fns(inner_dist)
    if relaxed:
        ts1d = derivative(ts1)
        ts2d = derivative(ts2)
    else:
        ts1d, ts2d = None, None
    dist = 0
    for point in path:
        i_f, i_t = point
        if not relaxed:
            dist += inner_dist(ts1[i_f], ts2[i_t])
            continue

        deriv_1, deriv_2 = ts1d[i_f], ts2d[i_t]
        ts1_1, ts1_2 = ts1[i_f] - deriv_1 * 0.5, ts1[i_f] + deriv_1 * 0.5
        ts2_1, ts2_2 = ts2[i_t] - deriv_2 * 0.5, ts2[i_t] + deriv_2 * 0.5
        dist += min(inner_dist(ts1_1, ts2_1),
                    inner_dist(ts1_1, ts2_2),
                    inner_dist(ts1_2, ts2_1),
                    inner_dist(ts1_2, ts2_2))
    return result(dist)


def get_points_in_path_with_min_diff(ts1, ts2, path, min_diff, relaxed=True):
    """Get all points in the path where the difference between the two
    points that are warped to each other is larger than the given
    minimum difference.

    For the relaxed version, see :method:`distance_from_path`.
    """
    points = []
    for point in path:
        i_f, i_t = point
        diff = abs(ts1[i_f] - ts2[i_t])
        if diff > min_diff:
            if not relaxed:
                points.append(point)
                continue
            # Be more relaxed about points with a high derivative, sampling has a large impact there
            if 0 < i_f < len(ts1) - 1:
                deriv_ref = ((ts1[i_f] - ts1[i_f - 1]) + (ts1[i_f + 1] - ts1[i_f - 1]) / 2) / 2
            else:
                deriv_ref = 1
            if 0 < i_t < len(ts2) - 1:
                deriv_seg = ((ts2[i_t] - ts2[i_t - 1]) + (ts2[i_t + 1] - ts2[i_t - 1]) / 2) / 2
            else:
                deriv_seg = 1
            ts1_1, ts1_2 = ts1[i_f] - deriv_ref * 0.5, ts1[i_f] + deriv_ref * 0.5
            ts2_1, ts2_2 = ts2[i_t] - deriv_seg * 0.5, ts2[i_t] + deriv_seg * 0.5
            diff = min(abs(ts1_1-ts2_1),
                       abs(ts1_1-ts2_2),
                       abs(ts1_2-ts2_1),
                       abs(ts1_2-ts2_2))
            if diff > min_diff:
                points.append(point)
    return points
