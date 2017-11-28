"""
dtaidistance.clustering - Clustering algorithms for Time Series

__author__ = "Wannes Meert"
__copyright__ = "Copyright 2017 KU Leuven, DTAI Research Group"
__license__ = "APL"

..
    Part of the DTAI distance code.

    Copyright 2017 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import logging
from collections import defaultdict
import numpy as np


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger("be.kuleuven.dtai.distance")


class Hierarchical:
    def __init__(self, dists_fun, dists_options, max_dist, weights=None, merge_hook=None):
        """

        :param dists_fun: Function to compute pairwise distance matrix between set of series.
        :param dists_options: Arguments to pass to dists_fun.
        :param max_dist: Do not merge or cluster series that are further apart than this.
        :param weights: Weight per series. Used to pick the prototype that defines a cluster.
            The clustering will try to group such that the prototypes have a high as possible summed weight.
        :param merge_hook: Function that is called when two series are clustered.
            The function definition is `def merge_hook(from_idx, to_idx)`, where idx is the index of the series.
        """
        self.dists_fun = dists_fun
        self.dists_options = dists_options
        self.weights = weights
        self.max_dist = max_dist
        self.merge_hook = merge_hook

    def fit(self, series):
        """Merge sequences.

        :param series: Iterator over series.
        :return: Dictionary with as keys the prototype indicices and as values all the indicides of the series in
            that cluster.
        """
        cluster_idx = dict()
        dists = self.dists_fun(series, **self.dists_options)
        min_value = np.min(dists)
        min_idxs = np.argwhere(dists == min_value)
        min_idx = -1
        max_cnt = 0
        if self.weights:
            for r, c in [min_idxs[ii, :] for ii in range(min_idxs.shape[0])]:
                total = self.weights[r] + self.weights[c]
                if total > max_cnt:
                    max_cnt = total
                    min_idx = (r, c)
        else:
            min_idx = min_idxs[0, :]
        deleted = set()
        cnt_merge = 0
        logger.info('Merging patterns')
        if tqdm:
            pbar = tqdm(total=dists.shape[0])
        else:
            pbar = None
        # Hierarchical clustering (distance to prototype)
        while min_value <= self.max_dist:
            cnt_merge += 1
            i1, i2 = int(min_idx[0]), int(min_idx[1])
            p1 = series[i1]
            p2 = series[i2]
            if self.weights:
                w1 = self.weights[i1]
                w2 = self.weights[i2]
                if w1 < w2 or (w1 == w2 and len(p1) > len(p2)):
                    i1, w1, i2, w2 = i2, w2, i1, w1
            logger.debug("Merge {} <- {} ({:.3f})".format(i1, i2, min_value))
            if i1 not in cluster_idx:
                cluster_idx[i1] = set([i1])
            if i2 in cluster_idx:
                cluster_idx[i1].update(cluster_idx[i2])
                del cluster_idx[i2]
            else:
                cluster_idx[i1].add(i2)

            if self.merge_hook:
                self.merge_hook(i2, i1)
            # if recompute:
            #     for r in range(i1):
            #         if r not in deleted and abs(len(cur_seqs[r]) - len(cur_seqs[i1])) <= max_length_diff:
            #             dists[r, i1] = self.dist(cur_seqs[r], cur_seqs[i1], **dist_opts)
            #     for c in range(i1+1, len(cur_seqs)):
            #         if c not in deleted and abs(len(cur_seqs[i1]) - len(cur_seqs[c])) <= max_length_diff:
            #             dists[i1, c] = self.dist(cur_seqs[i1], cur_seqs[c], **dist_opts)
            for r in range(i2):
                dists[r, i2] = np.inf
            for c in range(i2 + 1, len(series)):
                dists[i2, c] = np.inf
            deleted.add(i2)
            if pbar:
                pbar.update(1)
            # min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            # min_value = dists[min_idx]
            min_value = np.min(dists)
            min_idxs = np.argwhere(dists == min_value)
            min_idx = -1
            max_cnt = 0
            if self.weights:
                for r, c in [min_idxs[ii, :] for ii in range(min_idxs.shape[0])]:
                    total = self.weights[r] + self.weights[c]
                    if total > max_cnt:
                        max_cnt = total
                        min_idx = (r, c)
            else:
                min_idx = min_idxs[0, :]
        if pbar:
            pbar.update(dists.shape[0] - cnt_merge)

        prototypes = []
        for i in range(len(series)):
            if i not in deleted:
                prototypes.append(i)
                if i not in cluster_idx:
                    cluster_idx[i] = set(i)
        return cluster_idx
