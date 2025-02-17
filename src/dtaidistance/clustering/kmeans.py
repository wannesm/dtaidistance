# -*- coding: UTF-8 -*-
"""
dtaidistance.clustering.kmeans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(requires version 2.2.0 or higher)

Time series clustering using k-means and Barycenter averages.

:author: Wannes Meert
:copyright: Copyright 2020-2022 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import random
import math
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

from ..dtw import distance, distance_matrix_fast, distance_matrix, DTWSettings
from  .medoids import KMedoids
from ..util import SeriesContainer
from ..exceptions import NumpyException
from .medoids import Medoids
from ..dtw_barycenter import dba_loop
from .. import dtw, dtw_ndim


def _distance_with_params(t):
    series, avgs, dists_options = t
    min_i, min_d = -1, float('inf')
    for i, avg in enumerate(avgs):
        d = distance(series, avg, **dists_options)
        if d < min_d:
            min_d, min_i = d, i
    return min_i, min_d


def _distance_ndim_with_params(t):
    series, avgs, dists_options = t
    min_i, min_d = -1, float('inf')
    for i, avg in enumerate(avgs):
        d = dtw_ndim.distance(series, avg, **dists_options)
        if d < min_d:
            min_d, min_i = d, i
    return min_i, min_d


def _distance_c_with_params(t):
    series, means, dists_options = t
    min_i, min_d = -1, float('inf')
    for i, mean in enumerate(means):
        d = dtw_cc.distance(series, mean, **dists_options)
        if d < min_d:
            min_d, min_i = d, i
    return min_i, min_d


def _distance_ndim_c_with_params(t):
    series, means, dists_options = t
    min_i, min_d = -1, float('inf')
    for i, mean in enumerate(means):
        d = dtw_cc.distance_ndim(series, mean, **dists_options)
        if d < min_d:
            min_d, min_i = d, i
    return min_i, min_d


def _dba_loop_with_params(t):
    series, c, mask, max_it, thr, use_c, nb_prob_samples = t
    return dba_loop(series, c=c, mask=mask, max_it=max_it, thr=thr, use_c=use_c,
                    nb_prob_samples=nb_prob_samples)


class KMeans(Medoids):
    def __init__(self, k, max_it=10, max_dba_it=10, thr=0.0001, drop_stddev=None,
                 nb_prob_samples=None,
                 dists_options=None, show_progress=True,
                 initialize_with_kmedoids=False, initialize_with_kmeanspp=True,
                 initialize_sample_size=None):
        """K-means clustering algorithm for time series using Dynamic Barycenter
        Averaging.

        :param k: Number of components
        :param max_it: Maximal interations for K-means
        :param max_dba_it: Maximal iterations for the Dynamic Barycenter Averaging.
        :param thr: Convergence is achieved if the averaging iterations differ less
            than this threshold
        :param drop_stddev: When computing the average series per cluster, ignore
            the instances that are further away than stddev*drop_stddev from the
            prototype (this is a gradual effect, the algorithm starts with drop_stddev
            is 3).
            This is related to robust k-means approaches that use trimming functions.
        :param nb_prob_samples: Probabilistically sample best path this number of times.
        :param dists_options:
        :param show_progress:
        :param initialize_with_kmedoids: Cluster a sample of the dataset first using
            K-medoids.
        :param initialize_with_kmeanspp: Use k-means++
        :param initialize_sample_size: How many samples to use for initialization with K-medoids or K-means++.
            Defaults are k*20 for K-medoid and 2+log(k) for k-means++.
        """
        if dists_options is None:
            dists_options = {}
        self.means = [None] * k
        self.max_it = max_it
        self.max_dba_it = max_dba_it
        self.thr = thr
        self.drop_stddev = drop_stddev

        self.initialize_with_kmeanspp = initialize_with_kmeanspp
        self.initialize_with_kmedoids = initialize_with_kmedoids
        self.initialize_sample_size = initialize_sample_size
        self.nb_prob_samples = nb_prob_samples
        super().__init__(None, dists_options, k, show_progress)

    def kmedoids_centers(self, series):
        logger.debug('Start K-medoid initialization ... ')
        if self.initialize_sample_size is None:
            sample_size = min(self.k * 20, len(self.series))
        else:
            sample_size = min(self.initialize_sample_size, len(self.series))
        indices = np.random.choice(range(0, len(self.series)), sample_size, replace=False)
        sample = self.series[indices, :].copy()
        if self.dists_options.use_c:
            fn_dm = distance_matrix_fast
        else:
            fn_dm = distance_matrix
        model = KMedoids(fn_dm, {**self.dists_options, **{'compact': False}}, k=self.k)
        cluster_idx = model.fit(sample)
        means = [self.series[idx] for idx in cluster_idx.keys()]
        logger.debug('... Done')
        return means

    def kmeansplusplus_centers(self, series):
        """Better initialization for K-Means.

            Arthur, D., and S. Vassilvitskii. "k-means++: the, advantages of careful seeding.
            In, SODA'07: Proceedings of the." eighteenth annual ACM-SIAM symposium on Discrete, algorithms.

        Procedure (in paper):

        - 1a. Choose an initial center c_1 uniformly at random from X.
        - 1b. Choose the next center c_i , selecting c_i = x′∈X with probability D(x')^2/sum(D(x)^2, x∈X).
        - 1c. Repeat Step 1b until we have chosen a total of k centers.
        - (2-4. Proceed as with the standard k-means algorithm.)

        Extension (in conclusion):

        - Also, experiments showed that k-means++ generally performed better if it selected several new centers
          during each iteration, and then greedily chose the one that decreased φ as much as possible.

        Detail (in original code):

        - numLocalTries==2+log(k)

        :param series:
        :return:
        """
        if np is None:
            raise NumpyException("Numpy is required for the KMeans.kmeansplusplus_centers method.")
        logger.debug('Start K-means++ initialization ... ')
        ndim = self.series.detected_ndim
        if self.dists_options.get('use_c', False):
            if ndim == 1:
                fn = distance_matrix_fast
            else:
                fn = dtw_ndim.distance_matrix_fast
        else:
            if ndim == 1:
                fn = distance_matrix
            else:
                fn = dtw_ndim.distance_matrix
        indices = []
        if self.initialize_sample_size is None:
            n_samples = min(2 + int(math.log(self.k)), len(series) - self.k)
        else:
            n_samples = self.initialize_sample_size
        dists = np.empty((n_samples, len(series)))

        # First center is chosen randomly
        idx = np.random.randint(0, len(series))
        min_dists = np.power(fn(series, block=((idx, idx + 1), (0, len(series)), False),
                                compact=True, **self.dists_options), 2)
        indices.append(idx)

        for k_idx in range(1, self.k):
            # Compute the distance between each series and the nearest center that has already been chosen.
            # (Choose one new series at random as a new center, using a weighted probability distribution)
            # Select several new centers and then greedily chose the one that decreases pot as much as possible
            sum_min_dists = np.sum(min_dists)
            if sum_min_dists == 0.0:
                logger.warning('There are only {} < k={} different series'.format(k_idx, self.k))
                weights = None
            else:
                weights = min_dists / sum_min_dists
            idx_cand = np.random.choice(len(min_dists), size=n_samples, replace=False, p=weights)
            for s_idx, idx in enumerate(idx_cand):
                dists[s_idx, :] = np.power(fn(series, block=((idx, idx + 1), (0, len(series)), False),
                                              compact=True, **self.dists_options), 2)
                np.minimum(dists[s_idx, :], min_dists, out=dists[s_idx, :])
            potentials = np.sum(dists, axis=1)
            best_pot_idx = np.argmin(potentials)
            idx = idx_cand[best_pot_idx]
            min_dists[:] = dists[best_pot_idx, :]
            indices.append(idx)

        means = [series[i] for i in indices]
        assert(len(means) == self.k)
        logger.debug('... Done')
        return means

    def fit_fast(self, series, monitor_distances=None):
        use_c = self.dists_options.use_c
        self.dists_options.use_c = True
        result = self.fit(series, use_parallel=True, monitor_distances=monitor_distances)
        self.dists_options.use_c = use_c
        return result

    def fit(self, series, use_parallel=True, monitor_distances=None):
        """Perform K-means clustering.

        :param series: Container with series
        :param use_parallel: Use multipool for parallelization
        :param monitor_distances: This function is called with two arguments:
            (1) a list of (cluster, distance) for each instance;
            (2) a boolean indicating whether the clustering has been stopped or not.
            From this one can compute inertia or other metrics
            to monitor the clustering. If the boolean argument is true, this is the
            final assignment. If this function returns True, the clustering
            continues, if False is returned the clustering is stopped.
        :return: cluster indices, number of iterations
            If the number of iterations is equal to max_it, the clustering
            did not converge.
        """
        if np is None:
            raise NumpyException("Numpy is required for the KMeans.fit method.")
        self.series = SeriesContainer.wrap(series, support_ndim=True)
        ndim = self.series.detected_ndim
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

        if self.dists_options.get('use_c', False):
            if ndim == 1:
                fn = _distance_c_with_params
            else:
                fn = _distance_ndim_c_with_params
        else:
            if ndim == 1:
                fn = _distance_with_params
            else:
                fn = _distance_ndim_with_params

        # Initialisations
        if self.initialize_with_kmeanspp:
            self.means = self.kmeansplusplus_centers(self.series)
        elif self.initialize_with_kmedoids:
            self.means = self.kmedoids_centers(self.series)
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
            if monitor_distances is not None:
                cont = monitor_distances(clusters_distances, False)
                if cont is False:
                    break
            clusters, distances = zip(*clusters_distances)
            distances = list(distances)

            best_medoid = [0]*self.k
            best_dist = [float('inf')]*self.k
            for idx, (cluster, distance) in enumerate(clusters_distances):
                if distance < best_dist[cluster]:
                    best_dist[cluster] = distance
                    best_medoid[cluster] = idx

            if self.drop_stddev is not None and self.drop_stddev != 0:
                logger.debug('drop_stddev = {}'.format(drop_stddev))
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
            logger.debug('Ignored instances: {} / {} (max_value = {})'.format(cnt, len(clusters_distances), max_value))
            if (mask == mask_new).all():
                logger.info("Stopped after {} iterations, no change in cluster assignment".format(it_nb))
                break
            mask[:, :] = mask_new
            for ki in range(self.k):
                if not mask[ki, :].any():
                    idx = np.argmax(distances)
                    logger.debug('Empty cluster {}, assigned most dissimilar sequence {}={}'.format(ki, idx, distances[ki]))
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
                                    self.max_dba_it, self.thr, self.dists_options.get('use_c', False),
                                    self.nb_prob_samples)
                                   for ki in range(self.k)])
            else:
                means = list(map(_dba_loop_with_params,
                             [(self.series, self.series[best_medoid[ki]], mask[ki, :],
                               self.max_dba_it, self.thr, self.dists_options.get('use_c', False),
                               self.nb_prob_samples)
                              for ki in range(self.k)]))
            # for ki in range(self.k):
            #     means[ki] = dba_loop(self.series, c=None, mask=mask[:, ki], use_c=True)
            for ki in range(self.k):
                curlen = min(len(means[ki]), len(self.means[ki]))
                difflen += curlen
                if ndim == 1:
                    for a, b in zip(means[ki], self.means[ki]):
                        diff += abs(a - b)
                else:
                    for a, b in zip(means[ki], self.means[ki]):
                        diff += max(abs(a[d] - b[d]) for d in range(ndim))
                self.means[ki] = means[ki]
            diff /= difflen
            if diff <= self.thr:
                print("Stopped early after {} iterations, no change in means".format(it_nb))
                break

        # Final assignment
        if use_parallel:
            with mp.Pool() as p:
                clusters_distances = p.map(fn, [(self.series[idx], self.means, self.dists_options) for idx in
                                                range(len(self.series))])
        else:
            clusters_distances = list(map(fn, [(self.series[idx], self.means, self.dists_options) for idx in
                                               range(len(self.series))]))
        if monitor_distances is not None:
            monitor_distances(clusters_distances, True)
        clusters, distances = zip(*clusters_distances)

        # self.cluster_idx = {medoid: {inst for inst in instances}
        #                       for medoid, instances in zip(medoids, clusters)}
        self.cluster_idx = {ki: set() for ki in range(self.k)}
        for idx, cluster in enumerate(clusters):
            self.cluster_idx[cluster].add(idx)
        return self.cluster_idx, performed_it
