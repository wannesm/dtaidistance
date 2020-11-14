# -*- coding: UTF-8 -*-
"""
dtaidistance.kmeans
~~~~~~~~~~~~~~~~~~~

Time series clustering using k-means and Barycenter averages.

:author: Wannes Meert
:copyright: Copyright 2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import random
import math
from pathlib import Path
import multiprocessing as mp


logger = logging.getLogger("be.kuleuven.dtai.distance")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

dtw_cc = None
try:
    from .. import dtw_cc
except ImportError:
    logger.debug('DTAIDistance C library not available')
    dtw_cc = None


try:
    import numpy as np
except ImportError:
    np = None

from ..dtw import distance, distance_matrix_fast, distance_matrix
from  .medoids import KMedoids
from ..util import SeriesContainer
from ..exceptions import NumpyException, PyClusteringException, MatplotlibException
from .visualization import prepare_plot_options
from .medoids import Medoids
from ..dtw_barycenter import dba_loop


def _distance_with_params(t):
    series, avgs, dists_options = t
    min_i = -1
    min_d = float('inf')
    for i, avg in enumerate(avgs):
        d = distance(series, avg, **dists_options)
        if d < min_d:
            min_d = d
            min_i = i
    return min_i, min_d


def _distance_c_with_params(t):
    series, means, dists_options = t
    min_i = -1
    min_d = float('inf')
    for i, mean in enumerate(means):
        d = dtw_cc.distance(series, mean, **dists_options)
        if d < min_d:
            min_d = d
            min_i = i
    return min_i, min_d


def _dba_loop_with_params(t):
    series, c, mask, max_it, thr, use_c = t
    return dba_loop(series, c=c, mask=mask, max_it=max_it, thr=thr, use_c=use_c)


class KMeans(Medoids):
    def __init__(self, k, max_it=10, max_dba_it=10, thr=0.0001, drop_stddev=None,
                 dists_options=None, show_progress=True, initialize_with_kmedoids=True):
        """K-means clustering algorithm for time series using Dynamic Barycenter
        Averaging.

        :param k: Number of components
        :param max_it: Maximal interations for K-means
        :param max_dba_it: Maximal iterations for the Dynamic Barycenter Averaging.
        :param thr: Convergence is achieved if the averaging iterations differ less
            then this threshold
        :param drop_stddev: When computing the average series per cluster, ignore
            the instances that are further away than stddev*drop_stddev from the
            prototype (this is a gradual effect, the algorithm starts with drop_stddev
            is 3).
        :param dists_options:
        :param show_progress:
        :param initialize_with_kmedoids: Cluster a sample of the dataset first using
            K-medoids.
        """
        if dists_options is None:
            dists_options = {}
        dists_options['compact'] = False
        self.means = [None] * k
        self.max_it = max_it
        self.max_dba_it = max_dba_it
        self.thr = thr
        self.drop_stddev = drop_stddev
        self.initialize_with_kmedoids = initialize_with_kmedoids
        self.initialize_with_kmedoids_sample_size = k * 20
        super().__init__(None, dists_options, k, show_progress)

    def fit_fast(self, series):
        return self.fit(series, use_c=True, use_parallel=True)

    def fit(self, series, use_c=False, use_parallel=True):
        """Perform K-means clustering.

        :param series: Container with series
        :param use_c: Use the C-library (only available if package is compiled)
        :param use_parallel: Use multipool for parallelization
        :return: cluster indices, number of iterations
            If the number of iterations is equal to max_it, the clustering
            did not converge.
        """
        if np is None:
            raise NumpyException("Numpy is required for the KMeans.fit method.")
        self.series = SeriesContainer.wrap(series)
        mask = np.full((self.k, len(self.series)), False, dtype=bool)
        mask_new = np.full((self.k, len(self.series)), False, dtype=bool)
        means = [None] * self.k
        diff = 0
        performed_it = 1
        clusters = None
        if self.drop_stddev is not None:
            # Gradually tighten drop_stddev
            drop_stddev = max(self.drop_stddev, 4)
        else:
            drop_stddev = None

        if use_c:
            fn = _distance_c_with_params
        else:
            fn = _distance_with_params

        # Initialisations
        if self.initialize_with_kmedoids:
            logger.debug('Start K-medoid initialization ... ')
            sample_size = min(self.initialize_with_kmedoids_sample_size, len(self.series))
            indices = np.random.choice(range(0, len(self.series)), sample_size, replace=False)
            sample = self.series[indices, :].copy()
            if use_c:
                fn_dm = distance_matrix_fast
            else:
                fn_dm = distance_matrix
            model = KMedoids(fn_dm, self.dists_options, k=self.k)
            cluster_idx = model.fit(sample)
            self.means = [self.series[idx] for idx in cluster_idx.keys()]
            logger.debug('... Done')
        else:
            indices = np.random.choice(range(0, len(self.series)), self.k, replace=False)
            self.means = [self.series[random.randint(0, len(self.series) - 1)] for _ki in indices]

        # Iterations
        it_nbs = range(self.max_it)
        if self.show_progress and tqdm is not None:
            it_nbs = tqdm(it_nbs)
        for it_nb in it_nbs:

            # Assignment step
            performed_it += 1
            if use_parallel:
                with mp.Pool() as p:
                    clusters_distances = p.map(fn, [(self.series[idx], self.means, self.dists_options) for idx in
                                                    range(len(self.series))])
            else:
                clusters_distances = list(map(fn, [(self.series[idx], self.means, self.dists_options) for idx in
                                                   range(len(self.series))]))
            clusters, distances = zip(*clusters_distances)
            distances = list(distances)

            best_medoid = [0]*self.k
            best_dist = [float('inf')]*self.k
            for idx, (cluster, distance) in enumerate(clusters_distances):
                if distance < best_dist[cluster]:
                    best_dist[cluster] = distance
                    best_medoid[cluster] = idx

            if self.drop_stddev is not None and self.drop_stddev != 0:
                logger.debug(f'drop_stddev = {drop_stddev}')
                stats = []
                max_value = []
                for ki in range(self.k):
                    stats.append([0, 0, 0])
                for (cluster, distance) in clusters_distances:
                    stats[cluster][0] += distance
                    stats[cluster][2] += 1
                for ki in range(self.k):
                    if stats[ki][2] == 0:
                        stats[ki][0] = 0
                    else:
                        stats[ki][0] /= stats[ki][2]
                for (cluster, distance) in clusters_distances:
                    stats[cluster][1] += (stats[cluster][0] - distance)**2
                for ki in range(self.k):
                    if stats[ki][2] == 0:
                        stats[ki][1] = 0
                    else:
                        stats[ki][1] = math.sqrt(stats[ki][1]/stats[ki][2])
                    max_value.append(stats[ki][0] + stats[ki][1]*drop_stddev)
                drop_stddev = (drop_stddev + self.drop_stddev) / 2
            else:
                max_value = None

            mask_new[:, :] = False
            cnt = [0] * self.k
            for idx, (cluster, distance) in enumerate(clusters_distances):
                if max_value is None or distance <= max_value[cluster]:
                    # Ignore the far away instances to compute new averages.
                    mask_new[cluster, idx] = True
                else:
                    cnt[cluster] += 1
            logger.debug(f'Ignored instances: {cnt} / {len(clusters_distances)} (max_value = {max_value})')
            if (mask == mask_new).all():
                logger.info(f"Stopped after {it_nb} iterations, no change in cluster assignment")
                break
            mask[:, :] = mask_new
            for ki in range(self.k):
                if not mask[ki, :].any():
                    idx = np.argmax(distances)
                    logger.debug(f'Empty cluster {ki}, assigned most dissimilar sequence {idx}={distances[ki]}')
                    mask[:, idx] = False
                    mask[ki, idx] = True
                    distances[idx] = 0

            # Update step
            diff = 0
            difflen = 0
            if use_parallel:
                with mp.Pool() as p:
                    means = p.map(_dba_loop_with_params,
                                  [(self.series, self.series[best_medoid[ki]], mask[ki, :],
                                    self.max_dba_it, self.thr, use_c) for ki in range(self.k)])
            else:
                means = list(map(_dba_loop_with_params,
                             [(self.series, self.series[best_medoid[ki]], mask[ki, :],
                               self.max_dba_it, self.thr, use_c) for ki in range(self.k)]))
            # for ki in range(self.k):
            #     means[ki] = dba_loop(self.series, c=None, mask=mask[:, ki], use_c=True)
            for ki in range(self.k):
                curlen = min(len(means[ki]), len(self.means[ki]))
                difflen += curlen
                for a, b in zip(means[ki], self.means[ki]):
                    diff += abs(a - b)
                self.means[ki] = means[ki]
            diff /= difflen
            if diff <= self.thr:
                print(f"Stopped early after {it_nb} iterations, no change in means")
                break

        # self.cluster_idx = {medoid: {inst for inst in instances}
        #                       for medoid, instances in zip(medoids, clusters)}
        self.cluster_idx = {ki: set() for ki in range(self.k)}
        for idx, cluster in enumerate(clusters):
            self.cluster_idx[cluster].add(idx)
        return self.cluster_idx, performed_it


