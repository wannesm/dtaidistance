# -*- coding: UTF-8 -*-
"""
dtaidistance.subsequence.subsequencesearch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(requires version 2.3.0 or higher)

DTW-based subsequence matching.

:author: Wannes Meert
:copyright: Copyright 2021-2023 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging

from .. import dtw  # import warping_paths, warping_paths_fast, best_path, warping_paths_affinity, distance
from .. import dtw_ndim
from .. import util_numpy
from .. import util


try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
    import numpy.ma as ma
    argmin = np.argmin
    argmax = np.argmax
    array_min = np.min
    array_max = np.max
except ImportError:
    np = None
    ma = None
    argmin = util.argmin
    argmax = util.argmax
    array_min = min
    array_max = max


logger = logging.getLogger("be.kuleuven.dtai.distance")


dtw_cc = None
try:
    from . import dtw_cc
except ImportError:
    dtw_cc = None


def subsequence_search(query, series, dists_options=None, use_lb=True,
                       max_dist=None, max_value=None, use_c=None):
    """See SubsequenceSearch.

    :param query: Time series to search for
    :param series: Iterator over time series to perform search on.
            This can be for example windows over a long time series.
    :param dists_options: Options passed on to `dtw.distance`
    :param use_lb: Use lowerbounds to early abandon options
    :param max_dist: Ignore DTW distances larger than this value
    :param max_value: Ignore normalized DTW distances larger than this value
    :param use_c: Use fast C implementation if available
    :return: SubsequenceSearch object
    """
    ss = SubsequenceSearch(query, series, dists_options=dists_options, use_lb=use_lb,
                           max_dist=max_dist, max_value=max_value, use_c=use_c)
    return ss


class SSMatch:
    """Found match by SubsequenceSearch.

    The match is identified by the idx property, which is the index of the matched
    series in the original list of series. The distance property returns the DTW
    distance between the query and the series at index idx.
    """
    def __init__(self, kidx, ss):
        self.kidx = kidx
        self.ss = ss

    @property
    def distance(self):
        """DTW distance."""
        return self.ss.kbest_distances[self.kidx][0]

    @property
    def value(self):
        """Normalized DTW distance."""
        return self.distance / len(self.ss.query)

    @property
    def idx(self):
        return self.ss.kbest_distances[self.kidx][1]

    def __str__(self):
        return f'SSMatch({self.idx})'

    def __repr__(self):
        return self.__str__()


class SSMatches:
    def __init__(self, ss, k=None):
        """Iterator over matches.

        :param ss: The SubsequenceSearch object
        :param k: Optional a k. This overrules the ss.k value.
            Useful if a smaller k is asked to iterate over than has been stored.
        """
        self.ss = ss
        self.k = k
        if self.ss.kbest_distances is None:
            self.k = 0
        elif self.k is None or self.k > self.ss.k:
            self.k = self.ss.k
        if self.k is None:
            self.k = len(self.ss.kbest_distances)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            return [SSMatch(kip+start, self.ss) for kip, (_v, _i) in
                    enumerate(self.ss.kbest_distances[key])]
        return SSMatch(key, self.ss)

    def __iter__(self):
        for ki, (_v, _i) in enumerate(self.ss.kbest_distances[:self.k]):
            yield SSMatch(ki, self.ss)

    def __len__(self):
        return self.k

    def __str__(self):
        if self.k > 10:
            return '[' + ', '.join(str(m) for m in self[:5]) + ' ... ' +\
                   ', '.join(str(m) for m in self[-5:]) + ']'
        return '[' + ', '.join(str(m) for m in self) + ']'


class SubsequenceSearch:
    """
    :type distances: Optional[Iterable]
    """

    def __init__(self, query, s, dists_options=None, use_lb=True, keep_all_distances=False,
                 max_dist=None, max_value=None, use_c=None, use_ndim=None):
        """Search the best matching (subsequence) time series compared to a given time series.

        :param query: Time series to search for
        :param s: Iterator over time series to perform search on.
            This can be for example windows over a long time series.
        :param dists_options: Options passed on to `dtw.distance`
        :param use_lb: Use lowerbounds to early abandon options
        :param max_dist: Ignore DTW distances larger than this value
            if max_dist is also given in dists_options, then the one in dists_options is ignored
            if both max_dist and max_value are given, the smallest is used
        :param max_value: Ignore normalized DTW distances larger than this value
        """
        self.query = query
        if use_ndim is None:
            self.use_ndim = (util.detect_ndim(query) > 1)
        else:
            self.use_ndim = use_ndim
        self.s = s
        # If keep_all_distances is true, store all. Can take up quite some memory.
        self.distances = None
        # Keep track of the k-best distances
        self.kbest_distances = None
        self.lbs = None
        self.k = None
        self.dists_options = {} if dists_options is None else dists_options
        if max_dist is None:
            self.max_dist = self.dists_options.get('max_dist', np.inf)
        else:
            self.max_dist = max_dist
        if max_value is not None:
            self.max_dist = min(self.max_dist, max_value * len(self.query))
        self.dists_options['max_dist'] = self.max_dist
        if use_c is not None:
            self.dists_options['use_c'] = use_c
        self.use_lb = use_lb

        self.keep_all_distances = keep_all_distances
        # if self.use_lb and not self.keep_all_distances:
        #     raise ValueError("If argument use_lb is true, then keep_all_distances should also be true.")

    def reset(self):
        self.distances = None
        self.kbest_distances = None
        self.lbs = None

    # def compute_lbs(self):
    #     self.lbs = np.zeros((len(self.s),))
    #     for idx, series in enumerate(self.s):
    #         self.lbs[idx] = dtw.lb_keogh(self.query, series, **self.dists_options)

    def align_fast(self, k=None):
        use_c = self.dists_options['use_c']
        self.dists_options['use_c'] = True
        result = self.align(k=k)
        self.dists_options['use_c'] = use_c
        return result

    def align(self, k=None):
        if k is not None and self.k is not None and k <= self.k and self.kbest_distances is not None:
            return self.kbest_distances[:k]
        if self.use_ndim:
            distance = dtw_ndim.distance
            lb_keogh = None
            if self.use_lb:
                self.use_lb = False
                logger.warning('The setting use_lb is ignored for multivariate series.')
        else:
            distance = dtw.distance
            lb_keogh = dtw.lb_keogh
        if k is None or self.keep_all_distances:
            self.distances = np.zeros((len(self.s),))
            # if self.use_lb:
            #     self.compute_lbs()
        import heapq
        h = [(-np.inf, -1)]
        max_dist = self.max_dist
        self.dists_options['max_dist'] = max_dist
        for idx, series in enumerate(self.s):
            if self.use_lb:
                lb = lb_keogh(self.query, series, **self.dists_options)
                if lb > max_dist:
                    continue
            dist = distance(self.query, series, **self.dists_options)
            if k is not None:
                if len(h) < k:
                    if not np.isinf(dist) and dist <= max_dist:
                        heapq.heappush(h, (-dist, idx))
                        max_dist = min(max_dist, -h[0][0])
                else:
                    if not np.isinf(dist) and dist <= max_dist:
                        heapq.heappushpop(h, (-dist, idx))
                        max_dist = min(max_dist, -h[0][0])
                self.dists_options['max_dist'] = max_dist
            if self.keep_all_distances or k is None:
                self.distances[idx] = dist
        if k is not None:
            # hh = np.array([-v for v, _ in h])
            # self.kbest_distances = [(-h[i][0], h[i][1]) for i in np.argsort(hh)]
            self.kbest_distances = sorted((-v, i) for v, i in h if i != -1)
        else:
            self.kbest_distances = [(self.distances[i], i) for i in np.argsort(self.distances)]

        self.k = k
        return self.kbest_distances

    def get_ith_value(self, i):
        """Return the i-th value from the k-best values.

        :param i: Return i-th best value (i < k)
        :return: (distance, index)
        """
        if self.kbest_distances is None or self.k is None:
            raise ValueError('Align should be called before asking for the i-th value.')
        if i > self.k:
            raise ValueError('The i-th value is not available, i={}>k={}'.format(i, self.k))
        return self.kbest_distances[i]

    def best_match_fast(self):
        self.dists_options['use_c'] = True
        return self.best_match()

    def best_match(self):
        self.align(k=1)
        # _value, best_idx = self.kbest_distances[0]
        return SSMatch(0, self)

    def kbest_matches_fast(self, k=1):
        use_c = self.dists_options.get('use_c', None)
        self.dists_options['use_c'] = True
        result = self.kbest_matches(k=k)
        if use_c is None:
            del self.dists_options['use_c']
        else:
            self.dists_options['use_c'] = use_c
        return result

    def kbest_matches(self, k=1):
        """Return the k best matches.

        It is recommended to set k to a value, and not None.
        If k is set to None, all comparisons are kept and returned. Also, no early
        stopping is applied in case k is None.

        :param k: Number of best matches to return (default is 1)
        :return: List of SSMatch objects
        """
        self.align(k=k)
        # if k is None:
        #     return [SSMatch(best_idx, self) for best_idx in range(len(self.distances))]
        # if self.keep_all_distances:
        #     best_idxs = np.argpartition(self.distances, k)
        #     return [SSMatch(best_idx, self) for best_idx in best_idxs[:k]]
        # distances = reversed(sorted(self.h))
        # return [SSMatch(best_idx, self) for dist, best_idx in distances]
        return SSMatches(self)
