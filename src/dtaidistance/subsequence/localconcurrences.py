# -*- coding: UTF-8 -*-
"""
dtaidistance.subsequence.localconcurrences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(requires version 2.3.0 or higher)

DTW-based subsequence matching.

:author: Wannes Meert
:copyright: Copyright 2021-2023 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import math
from functools import partial

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


def local_concurrences(series1, series2=None, gamma=1, tau=0, delta=0, delta_factor=1, estimate_settings=None,
                       only_triu=None, penalty=None, window=None, use_c=False, compact=None):
    """Local concurrences, see LocalConcurrences.

    :param series1:
    :param series2:
    :param gamma: Affinity transformation exp(-gamma*(s1[i] - s2[j])**2)
    :param tau: threshold parameter
    :param delta: penalty parameter
        Should be negative. Added instead of the affinity score (if score below tau threshold parameter).
    :param delta_factor: multiply cumulative score (e.g. by 0.5).
        This is useful to have the same impact at different locations in the warping paths matrix, which
        is cumulative (and thus typically large in one corner and small in the opposite corner).
    :param estimate_settings: Estimate tau, delta, delta_factor from given series. Will be passed as
        tau_std to estimate_settings_from_std.
    :param only_triu: Only compute the upper traingle matrix values. Useful to avoid redundant computations
        when series1 is equal to series2 (or equivalently if series2 is None).
    :param penalty: Penalty that is added when dynamic programming is using moving vertically or horizontally
        through the matrix instead of diagonally. Used to prefer diagonal paths.
    :param compact: Use the compact representation for the warping paths matrix (only when use_c is true).
    :return:
    """
    lc = LocalConcurrences(series1, series2, gamma, tau, delta, delta_factor,
                           only_triu=only_triu, penalty=penalty, window=window, use_c=use_c, compact=compact)
    if estimate_settings is not None:
        lc.estimate_settings_from_std(series1, series2, tau_std=estimate_settings)
    lc.align()
    return lc


class LCMatch:
    def __init__(self, lc, row=None, col=None):
        """LocalConcurrences match"""
        self.row = row  # type: int
        self.col = col  # type: int
        self.lc = lc  # type: LocalConcurrences
        self._path = None

    @property
    def path(self):
        if self._path is not None:
            return self._path
        # TODO: always storing the path might be memory hungry
        #       but recomputing is impossible since the values are negated/masked afterwards
        self._path = self.lc.best_path(self.row, self.col)
        return self._path

    def distance(self, do_sqrt=True):
        if self._path is None:
            return None
        d = 0
        for r, c in self._path:
            d += (self.lc.series1[r] - self.lc.series2[c])**2
        if do_sqrt:
            d = math.sqrt(d)
        return d

    def __str__(self):
        return f'LCMatch({self.row, self.col})'

    def __repr__(self):
        return self.__str__()


class LCMatches:
    def __init__(self, lc,  matches=None):
        self._lc = lc
        self._matches = []
        if matches is not None:
            self._matches.update(matches)

    def __iter__(self):
        return self._matches.__iter__()

    def append(self, match):
        self._matches.append(match)

    def covered(self):
        s1 = np.zeros(len(self._lc.series1), dtype=np.bool)
        s2 = np.zeros(len(self._lc.series2), dtype=np.bool)
        for match in self._matches:
            path = match.path
            s1[path[0][0]:path[-1][0]+1] = True
            s2[path[0][1]:path[-1][1]+1] = True
        return s1, s2

    def coverage(self):
        s1, s2 = self.covered()
        c1 = np.sum(s1) / len(s1)
        c2 = np.sum(s2) / len(s2)
        return c1, c2

    def segments(self):
        s1, s2 = [], []
        for match in self._matches:
            p = match.path
            s1.append((p[0][0], p[-1][0]))
            s2.append((p[0][1], p[-1][1]))
        return s1, s2

    def missing(self):
        s1, s2 = self.coverage()
        s1 = ~s1
        s2 = ~s2
        return s1, s2

    def missing_segments(self):
        b1, b2 = self.covered()
        s1, s2 = self.segments()
        for sb, se in s1:
            b1[sb] = False
            b1[se] = False
        for sb, se in s2:
            b2[sb] = False
            b2[se] = False

        inmissing = False
        ms1 = []
        lstart = None
        for i in range(len(b1)):
            if inmissing:
                if b1[i]:
                    ms1.append((lstart, i))
                    inmissing = False
            else:
                if not b1[i]:
                    lstart = i
                    inmissing = True

        inmissing = False
        ms2 = []
        lstart = None
        for i in range(len(b2)):
            if inmissing:
                if b2[i]:
                    ms2.append((lstart, i))
                    inmissing = False
            else:
                if not b2[i]:
                    lstart = i
                    inmissing = True
        return ms1, ms2

    def distance(self, do_sqrt=True):
        d = 0
        for m in self._matches:
            d += m.distance(do_sqrt=False)
        if do_sqrt:
            d = math.sqrt(d)
        return d

    def distance_compensated(self, penalty=None, max_factor=10):
        """Distance with compensation for missed parts in sequences.

        :param penalty: Base penalty per missing step in the joint path
        :param max_factor: Number >1
        """
        if penalty is None:
            penalty = 1/self._lc.gamma
        d = self.distance(do_sqrt=False)
        c1, c2 = self.coverage()
        perc_missing = 1 - max(c1, c2)
        nb_missing = max((1-c1)*len(self._lc.series1), (1-c2)*len(self._lc.series2))
        if max_factor is not None:
            factor = 1 + ((max_factor - 1) * perc_missing)
        else:
            factor = 1
        d += factor * penalty * nb_missing
        d = math.sqrt(d)
        return d

    def plot(self, begin=None, end=None, showlegend=False, showpaths=True, showboundaries=True):
        from .. import dtw_visualisation as dtwvis
        if begin is None and end is None:
            series1 = self._lc.series1
            series2 = self._lc.series2
            wp = self._lc.wp_slice()
            begin = 0
        elif begin is None:
            series1 = self._lc.series1[:end]
            series2 = self._lc.series2[:end]
            wp = self._lc.wp_slice(re=end, ce=end)
            begin = 0
        elif end is None:
            series1 = self._lc.series1[begin:]
            series2 = self._lc.series2[begin:]
            wp = self._lc.wp_slice(rb=begin, cb=begin)
        else:
            series1 = self._lc.series1[begin:end]
            series2 = self._lc.series2[begin:end]
            wp = self._lc.wp_slice(rb=begin, re=end, cb=begin, ce=end)
        if begin is not None and begin > 0:
            includes_zero = False
        else:
            includes_zero = True
        fig, ax = dtwvis.plot_warpingpaths(series1, series2, wp, path=-1, showlegend=showlegend, includes_zero=includes_zero)
        if showpaths:
            nb_plotted = 0
            for i, match in enumerate(self._matches):
                path2 = []
                for t in match.path:
                    if begin is not None and (t[0] < begin or t[1] < begin):
                        continue
                    if end is not None and (t[0] > (end-1) or t[1] > (end-1)):
                        continue
                    path2.append((t[0] - begin, t[1] - begin))
                if len(path2) > 0:
                    nb_plotted += 1
                    dtwvis.plot_warpingpaths_addpath(ax, path2)
            print(f"Paths plotted: {nb_plotted}")
        if showboundaries:
            # s1, s2 = self.covered()
            ss1, ss2 = self.segments()
            sbs, ses = zip(*ss1)
            sbs1 = [v - begin for v in sbs if (begin is None or v >= begin) and (end is None or v <= end)]
            ses1 = [v - begin for v in ses if (begin is None or v >= begin) and (end is None or v <= end)]
            ax[3].hlines(sbs1, 0, len(series2) - 1, color='black', alpha=0.5)
            ax[3].hlines(ses1, 0, len(series2) - 1, color='black', alpha=0.5)
            sbs, ses = zip(*ss2)
            sbs2 = [v - begin for v in sbs if (begin is None or v >= begin) and (end is None or v <= end)]
            ses2 = [v - begin for v in ses if (begin is None or v >= begin) and (end is None or v <= end)]
            ax[3].vlines(sbs2, 0, len(series1) - 1, color='black', alpha=0.5)
            ax[3].vlines(ses2, 0, len(series1) - 1, color='black', alpha=0.5)
            ymin = min(np.min(series1), np.min(series2))
            for idx, (sb, se) in enumerate(zip(sbs1, ses1)):
                ax[2].plot([-ymin, -ymin], [len(series1)-sb, len(series1)-se], color='blue', linewidth=2, alpha=0.5)
            for idx, (sb, se) in enumerate(zip(sbs2, ses2)):
                ax[1].plot([sb, se], [ymin, ymin], color='blue', linewidth=2, alpha=0.5)
        return fig, ax

    def str(self, maxlength=10):
        return '[' + ', '.join(str(m) for m in self._matches[:maxlength]) + ']'

    def __str__(self):
        return self.str()


class LocalConcurrences:
    def __init__(self, series1, series2=None, gamma=1, tau=0, delta=0, delta_factor=1, only_triu=False,
                 penalty=None, window=None, use_c=False, compact=None):
        """Version identification based on local concurrences.

        Find recurring patterns across two time series. Used to identify whether one time series is
        a version of another. If the two time series are the same one, it can be used to find typical
        or frequent patterns in a time series.

        Based on 7.3.2 Identiﬁcation Procedure in Fundamentals of Music Processing, Meinard Müller, Springer, 2015.

        Different from the original formulation, D_tau is introduced based on the given delta factor.
        This makes the penalty less sensitive to the cumulative effect of the paths in the
        self-similarity matrix S:

        S_tau(n,m) = S(n,m)  if  S(n,m) >= tau  (with tau >= 0)
                     delta   if  S(n,m) < tau   (with tau >= 0 & delta <= 0)

        And for the accumulated score matrix D:

        D_tau(n,m) = max(0,
                         df * D_tau(n−1,m−1) + S_tau(n,m),
                         df * D_tau(n−1,m)   + S_tau(n,m),
                         df * D_tau(n,m−1)   + S_tau(n,m))
        where df = 1 if S(n,m) >= tau and df=delta_factor (<=1) otherwise,

        For finding paths the delta_factor has no influence. For the visualisation,
        it helps as patterns exhibit more similar values in the D matrix.

        :param series1: First time series.
        :param series2: Second time series. If empty, series1 is used and compared with itself.
        :param gamma: Affinity transformation exp(-gamma*(s1[i] - s2[j])**2), should be >0
        :param tau: threshold parameter, should be >= 0
        :param delta: penalty parameter, should be <= 0
        :param delta_factor: penalty factor parameter, should be <= 1
        :param only_triu: Only consider upper triangular matrix in warping paths.
        :param compact: Use the compact representation for the warping paths matrix (only when use_c is true).
        """
        self.series1 = series1
        if series2 is None:
            # Self-comparison
            self.series2 = self.series1
            self.only_triu = True if only_triu is None else only_triu
        else:
            self.series2 = series2
            self.only_triu = False if only_triu is None else only_triu
        self.gamma = gamma
        self.tau = tau
        self.delta = delta
        self.delta_factor = delta_factor
        self.penalty = penalty
        self.window = window
        self.use_c = use_c
        if compact is None:
            self.compact = self.use_c
        else:
            self.compact = compact
        self._wp = None  # warping paths
        if self.use_c:
            self._c_settings = dtw_cc.DTWSettings(window=self.window, penalty=self.penalty)
            self._c_parts = dtw_cc.DTWWps(len(self.series1), len(self.series2), self._c_settings)

    @staticmethod
    def from_other(lc, series1, series2=None):
        lcn = LocalConcurrences(series1, series2, gamma=lc.gamma, tau=lc.tau, delta=lc.delta,
                                delta_factor=lc.delta_factor, only_triu=lc.only_triu,
                                penalty=lc.penalty, window=lc.window, use_c=lc.use_c, compact=lc.compact)
        return lcn

    def reset(self):
        self._wp = None

    def estimate_settings_from_std(self, series, series2=None, tau_std=0.33):
        """Estimate delta, tau and delta_factor from series, tau_std and gamma.

        :param series:
        :param tau_std: Set tau to differences larger than tau_std time standard deviation of
            the given series (default is 0.33, or reject differences that are larger than
            the deviation wrt to the mean of 75% of the values in the series, assuming a
            normal distribution).
        :return:
        """
        return self.estimate_settings(series, series2, tau_type='std', tau_factor=tau_std)

    def estimate_settings_from_mean(self, series, series2=None, tau_mean=0.33):
        return self.estimate_settings(series, series2, tau_type='mean', tau_factor=tau_mean)

    def estimate_settings_from_abs(self, series, series2=None, tau_abs=0.33):
        return self.estimate_settings(series, series2, tau_type='abs', tau_factor=tau_abs)

    def estimate_settings(self, series, series2=None, tau_factor=0.33, tau_type='mean', gamma=None):
        if tau_type != 'abs':
            if series is None:
                diffm = 1
            elif series2 is None:
                if tau_type == 'std':
                    diffm = np.std(series)
                elif tau_type == 'mean':
                    diffm = np.mean(series)
                else:
                    diffm = 1
            else:
                if tau_type == 'std':
                    diffm = np.std(np.abs(series - series2))
                elif tau_type == 'mean':
                    diffm = np.mean(np.abs(series - series2))
                else:
                    diffm = 1

            if gamma is None:
                # Intuition for gamma:
                # Create an affinity matrix where
                # differences up to the mean/std are in [e^-1, 1],
                # larger differences are i [0, e^-1]
                self.gamma = 1 / diffm**2
            else:
                self.gamma = gamma
            if tau_factor is not None:
                diffp = tau_factor * diffm
            else:
                diffp = diffm
        elif tau_type == 'abs':
            diffp = tau_factor
        else:
            raise AttributeError('{} is not supported (not in mean, std, abs)'.format(tau_type))
        self.tau = np.exp(-self.gamma * diffp ** 2)
        self.delta = -2 * self.tau
        self.delta_factor = 0.90
        self.penalty = self.tau / 10

    def align(self):
        """

        :return:
        """
        if self._wp is not None:
            return
        if self.use_c:
            fn = partial(dtw.warping_paths_affinity_fast, compact=self.compact)
        else:
            fn = dtw.warping_paths_affinity
        _, wp = fn(self.series1, self.series2,
                   gamma=self.gamma, tau=self.tau, delta=self.delta, delta_factor=self.delta_factor,
                   only_triu=self.only_triu, penalty=self.penalty, window=self.window)
        if self.compact:
            self._wp = wp
        else:
            self._wp = ma.masked_array(wp)
        self._reset_wp_mask()
        # if self.only_triu:
        #     il = np.tril_indices(self._wp.shape[0])
        #     self._wp[il] = ma.masked

    def align_fast(self):
        use_c = self.use_c
        self.use_c = True
        result = self.align()
        self.use_c = use_c
        return result

    def _reset_wp_mask(self):
        if self.compact:
            dtw_cc.wps_positivize(self._c_parts, self._wp,
                                  len(self.series1), len(self.series2),
                                  0, len(self.series1) + 1,
                                  0, len(self.series2) + 1)
        else:
            wp = self._wp
            if self.window is None:
                wp.mask = False
            else:
                windowdiff1 = max(0, wp.shape[1] - wp.shape[0])
                windowdiff2 = max(0, wp.shape[0] - wp.shape[1])
                il = np.tril_indices(n=wp.shape[0], k=-1 - self.window - windowdiff2, m=wp.shape[1])
                wp[il] = ma.masked
                il = np.triu_indices(n=wp.shape[0], k=-self.window - windowdiff2, m=wp.shape[1])
                wp.mask[il] = False
                il = np.triu_indices(n=wp.shape[0], k=1 + self.window + windowdiff1, m=wp.shape[1])
                wp[il] = ma.masked
            if self.only_triu:
                il = np.tril_indices(self._wp.shape[0], k=-1)
                wp[il] = -np.inf
                wp[il] = ma.masked

    def similarity_matrix(self):
        sm = ma.masked_array(np.empty((len(self.series1), len(self.series2))))
        for r in range(len(self.series1)):
            if self.window is None:
                minc, maxc = 0, len(self.series2)
            else:
                minc, maxc = max(0, r - self.window), min(len(self.series2), r + self.window)
            for c in range(minc):
                sm[r, c] = ma.masked
            for c in range(minc, maxc):
                d = np.exp(-self.gamma * (self.series1[r] - self.series2[c]) ** 2)
                sm[r, c] = self.delta if d < self.tau else d
            for c in range(maxc, len(self.series2)):
                sm[r, c] = ma.masked
        return sm

    def similarity_matrix_matshow_kwargs(self, sm):
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm
        # viridis = cm.get_cmap('viridis', 256)
        # newcolors = viridis(np.linspace(0, 1, 256))
        # pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
        # newcolors[:25, :] = pink
        # newcmp = ListedColormap(newcolors)
        # define the colormap
        cmap = plt.get_cmap('Spectral')
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        # define the bins and normalize and forcing 0 to be part of the colorbar!
        sm[sm == -np.inf] = 0
        sm_max = max(1, np.max(sm))
        sm_min = min(self.delta, np.min(sm))
        # bounds_pos = np.arange(0, sm_max, .01)
        bounds_pos = np.linspace(0, sm_max, 128)
        bounds_neg = np.linspace(sm_min, 0, len(bounds_pos))
        bounds = np.concatenate((bounds_neg, bounds_pos))
        # bounds = np.arange(sm_min, sm_max, .01)
        # idx = np.searchsorted(bounds, 0)
        # bounds = np.insert(bounds, idx, 0)
        norm = BoundaryNorm(bounds, cmap.N)
        return {'cmap': cmap, 'norm': norm}

    @property
    def wp(self):
        if self.compact:
            raise NotImplementedError("The full warping paths matrix is not available when using compact=True.\n"
                                      "Use wp_slice to construct part of the matrix from the compact data structure.")
        return self._wp.data

    def wp_slice(self, rb=None, re=None, cb=None, ce=None, positivize=False):
        if rb is None:
            rb = 0
        if re is None:
            re = len(self.series1) + 1
        if cb is None:
            cb = 0
        if ce is None:
            ce = len(self.series2) + 1
        if not (0 <= rb <= len(self.series1) + 1 and
                0 <= re <= len(self.series1) + 1 and
                0 <= cb <= len(self.series2) + 1 and
                0 <= ce <= len(self.series2) + 1):
            raise ValueError('Slice needs to be in 0<=r<={} and 0<=c<={}'.format(len(self.series1) + 1,
                                                                                 len(self.series2) + 1))
        if self.compact:
            slice = np.empty((re-rb, ce-cb), dtype=np.double)
            dtw_cc.wps_expand_slice(self._wp, slice, len(self.series1), len(self.series2),
                                    rb, re, cb, ce, self._c_settings)
        else:
            slice = self._wp[rb:re, cb:ce]
        if positivize:
            neg_idx = slice < 0
            slice[neg_idx] = -slice[neg_idx]
        return slice

    def best_match(self):
        idx = np.unravel_index(np.argmax(self._wp, axis=None), self._wp.shape)
        r, c = idx
        lcm = LCMatch(self, r, c)
        # path = lcm.path
        # for (x, y) in path:
        #     self._wp[x + 1, y + 1] = ma.masked
        return lcm

    def kbest_matches_store(self, k=1, minlen=2, buffer=0, restart=True, keep=False, matches=None, tqdm=None):
        import time
        if matches is None:
            matches = LCMatches(self)
        it = self.kbest_matches(k=k, minlen=minlen, buffer=buffer, restart=restart)
        if tqdm is not None:
            it = tqdm(it, total=k)
        tp = time.perf_counter()
        for ki, match in enumerate(it):
            matches.append(match)
            tn = time.perf_counter()
            #print(f'time: {tn-tp}')
            tp = tn
        if not keep:
            self._reset_wp_mask()
        return matches

    def kbest_matches(self, k=1, minlen=2, buffer=0, restart=True):
        """Yields the next best LocalConcurrent match.
        Stops at k matches (use None for all matches).

        :param k: Number of matches to yield, None is all matches
        :param minlen: Consider only matches of length longer than minlen
        :param buffer: Matches cannot be closer than buffer to each other
        :param restart: Start searching from start, ignore previous calls to kbest_matches
        :param keep: Keep mask to search incrementally for multiple calls of kbest_matches
        :return: Yield an LCMatch object
        """
        if self._wp is None:
            self.align()
        wp = self._wp
        if restart:
            self._reset_wp_mask()
        l1 = len(self.series1)
        l2 = len(self.series2)
        lperc = max(100, int(l1/10))
        ki = 0
        while k is None or ki < k:
            idx = None
            lcm = None
            cnt = 0
            while idx is None:
                cnt += 1
                if cnt % lperc == 0:
                    print(f'Searching for matches is taking a long time (k={ki+1}/{k}: {cnt} tries)')
                if self.compact:
                    idx = dtw_cc.wps_max(self._c_parts, wp, l1, l2)
                else:
                    idx = np.unravel_index(np.argmax(wp, axis=None), wp.shape)
                if idx[0] == 0 or idx[1] == 0:
                    # If all are masked, idx=0 is returned
                    return None
                r, c = idx
                # print(f'Best value: wp[{r},{c}] = {wp[r,c]}')
                lcm = LCMatch(self, r, c)
                path = lcm.path
                for (x, y) in path:
                    x += 1
                    y += 1
                    if not self.compact:
                        if len(wp.mask.shape) > 0 and wp.mask[x, y] is True:  # True means invalid
                            # print('found path contains masked, restart')
                            lcm = None
                            idx = None
                            break
                        else:
                            wp[x, y] = -wp[x, y]  # ma.masked
                    else:
                        dtw_cc.wps_negativize_value(self._c_parts, wp, l1, l2, x, y)
                if len(path) < minlen:
                    # print('found path too short, restart')
                    lcm = None
                    idx = None
            if buffer < 0 and lcm is not None:
                if self.compact:
                    dtw_cc.wps_negativize(self._c_parts, wp,
                                          len(self.series1), len(self.series2),
                                          path[0][0]+1, path[-1][0]+2,
                                          path[0][1]+1, path[-1][1]+2,
                                          True)  # intersection
                else:
                    miny, maxy = 0, wp.shape[1]
                    minx, maxx = 0, wp.shape[0]
                    wp[path[0][0]+1:path[-1][0]+2, miny:maxy] = -wp[path[0][0]+1:path[-1][0]+2, miny:maxy]  # ma.masked
                    wp[minx:maxx, path[0][1]+1:path[-1][1]+2] = -wp[minx:maxx, path[0][1]+1:path[-1][1]+2]  # ma.masked
            elif buffer > 0 and lcm is not None:
                miny, maxy = 0, wp.shape[1] - 1
                minx, maxx = 0, wp.shape[0] - 1
                if self.compact:
                    raise Exception("A positive buffer is not yet supported for compact WP data structure")
                else:
                    for (x, y) in path:
                        xx = x + 1
                        for yy in range(max(miny, y + 1 - buffer), min(maxy, y + 1 + buffer)):
                            wp[xx, yy] = -wp[xx, yy]  # ma.masked
                        yy = y + 1
                        for xx in range(max(minx, x + 1 - buffer), min(maxx, x + 1 + buffer)):
                            wp[xx, yy] = -wp[xx, yy]  # ma.masked
            if lcm is not None:
                ki += 1
                yield lcm

    def best_path(self, row, col, wp=None):
        if self._wp is None:
            return None
        if wp is None:
            wp = self._wp
        l1 = len(self.series1)
        l2 = len(self.series2)
        if self.compact:
            p = dtw_cc.best_path_compact_affinity(wp, l1, l2, row, col, window=self.window)
            return p
        argm = argmax
        i = row
        j = col
        p = [(i - 1, j - 1)]
        # prev = self._wp[i, j]
        while i > 0 and j > 0:
            values = [wp[i - 1, j - 1], wp[i - 1, j], wp[i, j - 1]]
            # print(f'{i=}, {j=}, {argm(values)=}, {ma.argmax(values)=}, {values=}')
            values = [-1 if v is ma.masked else v for v in values]
            c = argmax(values)  # triggers "Warning: converting a masked element to nan"
            # if values[c] is ma.masked:
            #     break
            if values[c] <= 0:  # values[c] > prev:
                break
            # prev = values[c]
            if c == 0:
                if wp[i - 1, j - 1] is ma.masked or wp[i - 1, j - 1] < 0:
                    assert False
                    break
                i, j = i - 1, j - 1
            elif c == 1:
                if wp[i - 1, j] is ma.masked or wp[i - 1, j] < 0:
                    assert False
                    break
                i = i - 1
            elif c == 2:
                if wp[i, j - 1] is ma.masked or wp[i, j - 1] < 0:
                    assert False
                    break
                j = j - 1
            p.append((i - 1, j - 1))
        if p[-1][0] < 0 or p[-1][1] < 0:
            p.pop()
        p.reverse()
        return p

    def settings_from(self, lc):
        self.gamma = lc.gamma
        self.tau = lc.tau
        self.delta = lc.delta
        self.delta_factor = lc.delta_factor
        self.penalty = lc.penalty
        self.window = lc.window

    def settings(self, kind=None):
        d = {
            "gamma": self.gamma,
            "tau": self.tau,
            "delta": self.delta,
            "delta_factor": self.delta_factor,
            "penalty": self.penalty,
            "window": self.window,
        }
        if kind == "str":
            return "\n".join(f"{k:<13}: {v}" for k, v in d.items())
        return d

    def wp_c_print(self):
        dtw_cc.wps_print(self._wp, len(self.series1), len(self.series2), window=self.window)

    def wp_c_print_compact(self):
        dtw_cc.wps_print_compact(self._wp, len(self.series1), len(self.series2), window=self.window)
