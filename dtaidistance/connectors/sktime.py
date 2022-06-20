# -*- coding: UTF-8 -*-
"""
dtaidistance.sktime
~~~~~~~~~~~~~~~~~~~

Connectors to use DTAIDistance with sktime

:author: Wannes Meert
:copyright: Copyright 2022 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
from .. import dtw
import numpy as np


def dtw_distance(x: np.ndarray, y: np.ndarray, dtw_settings=None, **kwargs) -> float:
    """Function compatible with sklearn.

    Can be used like:
    ```
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    from dtaidistance.connectors.sktime import dtw_distance
    KNeighborsTimeSeriesClassifier(distance=dtw_distance)
    ```
    """
    if dtw_settings is None:
        dtw_settings = {}
    # DTAIDistance expact a row of values, sktime expects a column
    x = x[:, 0]
    y = y[:, 0]
    d = dtw.distance(x, y, **dtw_settings)
    return d


def dtw_distance_create(**hmtw_settings):
    def dtw_distance_fn(x, y, **kwargs):
        return dtw_distance(x, y, hmtw_settings, **kwargs)
    return dtw_distance_fn
