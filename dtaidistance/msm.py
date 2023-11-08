# -*- coding: UTF-8 -*-
"""
dtaidistance.msm
~~~~~~~~~~~~~~~~

Move-Split-Merge (MSM)

:author: Wannes Meert
:copyright: Copyright 2022 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import math

import numpy as np


def distance(x, y, sm_cost=0.1):
    """MSM distance

    A. Stefan, V. Athitsos, and G. Das.
    The move-split-merge metric for time series.
    IEEE transactions on Knowledge and Data Engineering,
    25(6):1425–1438, 2012.

    :param x: first time series
    :param y: second time series
    :param sm_cost: Split-Merge cost
    :return: MSM distance
    """
    # setup
    def c(a, b, c):
        if (b <= a <= c) or (b >= a >= c):
            return sm_cost
        return sm_cost + min(abs(a - b), abs(a - c))
    m = len(x)
    n = len(y)
    cost = np.zeros((m, n))

    # initialization
    cost[0, 0] = abs(x[0] - y[0])
    for i in range(1, m):
        cost[i, 0] = cost[i-1, 0] + c(x[i], x[i-1], y[0])
    for j in range(1, n):
        cost[0, j] = cost[0, j-1] + c(y[j], x[0], y[j-1])

    # main loop
    for i in range(1, n):
        for j in range(1, m):
            d = [cost[i - 1][j - 1] + abs(x[i] - y[j]),
                 cost[i - 1][j] + c(x[i], x[i - 1], y[j]),
                 cost[i][j - 1] + c(y[j], x[i], y[j - 1])]
            # print(d)
            cost[i, j] = np.min(d)

    return cost[m-1, n-1]
