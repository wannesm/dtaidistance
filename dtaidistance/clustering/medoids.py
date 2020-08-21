# -*- coding: UTF-8 -*-
"""
dtaidistance.medoids
~~~~~~~~~~~~~~~~~~~~

Time series clustering.

:author: Wannes Meert
:copyright: Copyright 2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import math
import logging
from pathlib import Path
from collections import deque

try:
    import numpy as np
except ImportError:
    np = None

from ..util import SeriesContainer
from ..exceptions import NumpyException, MatplotlibException, ScipyException


logger = logging.getLogger("be.kuleuven.dtai.distance")


class KMedoids:
    def __init__(self, dists_fun, dists_options, k, show_progress=True):
        """

        Based on PAM.
        Schubert, Erich, and Peter J. Rousseeuw. "Faster k-medoids clustering: improving the PAM, CLARA, and
        CLARANS algorithms." International Conference on Similarity Search and Applications. Springer, Cham, 2019.

        :param dists_fun:
        :param dists_options:
        :param show_progress:
        """
        self.dists_fun = dists_fun
        self.dists_options = dists_options
        self.show_progress = show_progress
        self.k = k

    def fit(self, series):
        if np is None:
            raise NumpyException("The fit function requires Numpy to be installed.")
        nb_series = len(series)
        dists = self.dists_fun(series, **self.dists_options)
        # Make the matrix symmetric
        i_lower = np.tril_indices(nb_series, -1)
        dists[i_lower] = dists.T[i_lower]
        dists[np.isinf(dists)] = 0
        # Initial clusters
        medoids = self.build_initial_clustering(dists)
        print(f'Initial clusters: {medoids}')


    def build_initial_clustering(self, dists):
        medoids = []
        sumdiss = np.sum(dists, axis=1)
        smallest_idx = np.argmin(sumdiss)
        print(smallest_idx)
        medoids.append(smallest_idx)
        for k_i in range(self.k - 1):

        return medoids


