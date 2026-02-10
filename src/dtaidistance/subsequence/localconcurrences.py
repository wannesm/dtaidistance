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
import time

from .. import dtw
from .. import util_numpy
from .. import util
from ..innerdistance import StepsType
from ..exceptions import NumpyException


try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
    import numpy.ma as ma
    argmin = np.argmin
    argmax = np.argmax
    array_min = np.min
    array_max = np.max
    DTYPE = util_numpy.seq_t_dtype
except ImportError:
    np = None
    ma = None
    argmin = util.argmin
    argmax = util.argmax
    array_min = min
    array_max = max
    DTYPE = None


logger = logging.getLogger("be.kuleuven.dtai.distance")
inf = float("inf")
steps_type_default = "TypeIII"


dtw_cc = None
try:
    from .. import dtw_cc
except ImportError:
    dtw_cc = None

loco_cc = None
try:
    from .. import loco_cc
except ImportError:
    loco_cc = None


def local_concurrences(series1, series2=None, gamma=1, tau=0, delta=0, delta_factor=1, estimate_settings=None,
                       only_triu=None, penalty=None, window=None, use_c=False, steps_type=None):
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
    :return:
    """
    lc = LocalConcurrences(series1, series2, gamma, tau, delta, delta_factor,
                           only_triu=only_triu, penalty=penalty, window=window, use_c=use_c,
                           steps_type=steps_type)
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
        self._lc = lc  # type: LocalConcurrences
        self._matches = []  # type: list[LCMatch]
        if matches is not None:
            self._matches.update(matches)

    def __iter__(self):
        return self._matches.__iter__()

    def __len__(self):
        return self._matches.__len__()

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

    def plot(self, begin=None, end=None, showlegend=False, showpaths=True, showboundaries=True,
             makepositive=True, showpathidx=False, figure=None, **kwargs):
        """

        :param begin: Slice with this start index of time series
        :param end: Slice with this end index of time series
        :param showlegend:
        :param showpaths:
        :param showboundaries:
        :param makepositive:
        :param showpathidx:
        :param figure:
        :param kwargs:
        :return:
        """
        from .. import dtw_visualisation as dtwvis
        if begin is None and end is None:
            series1 = self._lc.series1
            series2 = self._lc.series2
            wp = self._lc.wp_slice_ts()
            begin = 0
        elif begin is None:
            series1 = self._lc.series1[:end]
            series2 = self._lc.series2[:end]
            wp = self._lc.wp_slice_ts(re=end, ce=end)
            begin = 0
        elif end is None:
            series1 = self._lc.series1[begin:]
            series2 = self._lc.series2[begin:]
            wp = self._lc.wp_slice_ts(rb=begin, cb=begin)
        else:
            series1 = self._lc.series1[begin:end]
            series2 = self._lc.series2[begin:end]
            wp = self._lc.wp_slice_ts(rb=begin, re=end, cb=begin, ce=end)
        if makepositive:
            wp = wp.copy()
            wp_neg = wp < 0
            wp[wp_neg] = -wp[wp_neg]
        fig, ax = dtwvis.plot_warpingpaths(series1, series2, wp, path=-1,
                                           showlegend=showlegend,
                                           includes_zero=False,
                                           figure=figure, **kwargs)
        ax_wps = ax[0]
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
                    if showpathidx:
                        ax_wps.text(path2[0][1], path2[0][0], str(i), color='red',
                                    horizontalalignment='right', verticalalignment='top')
            print(f"Paths plotted: {nb_plotted}")
        if showboundaries:
            ymin = min(np.min(series1), np.min(series2))
            ss1, ss2 = self.segments()
            if len(ss1) > 0:
                # Left time series
                sbs, ses = zip(*ss1)
                sbs1 = [v - begin for v in sbs if (begin is None or v >= begin) and (end is None or v <= end)]
                ses1 = [v - begin for v in ses if (begin is None or v >= begin) and (end is None or v <= end)]
                ax_wps.hlines(sbs1, 0, len(series2) - 1, color='black', alpha=0.25)
                ax_wps.hlines(ses1, 0, len(series2) - 1, color='black', alpha=0.25)
                for idx, (sb, se) in enumerate(zip(sbs1, ses1)):
                    ax[2].plot([-ymin, -ymin], [len(series1) - sb - 1, len(series1) - se - 1], color='blue', linewidth=2, alpha=0.5)
            if len(ss2) > 0:
                # Top time series
                sbs, ses = zip(*ss2)
                sbs2 = [v - begin for v in sbs if (begin is None or v >= begin) and (end is None or v <= end)]
                ses2 = [v - begin for v in ses if (begin is None or v >= begin) and (end is None or v <= end)]
                ax_wps.vlines(sbs2, 0, len(series1) - 1, color='black', alpha=0.25)
                ax_wps.vlines(ses2, 0, len(series1) - 1, color='black', alpha=0.25)
                for idx, (sb, se) in enumerate(zip(sbs2, ses2)):
                    ax[1].plot([sb, se], [ymin, ymin], color='blue', linewidth=2, alpha=0.5)
        return fig, ax

    def str(self, maxlength=10):
        return '[' + ', '.join(str(m) for m in self._matches[:maxlength]) + ']'

    def __str__(self):
        return self.str()


class LocalConcurrences:
    def __init__(self, series1, series2=None, gamma=1, tau=0, delta=0, delta_factor=1, only_triu=False,
                 penalty=0, window=None, use_c=False, steps_type=None):
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
        self.steps_type = StepsType.wrap(steps_type, steps_type_default)
        self.steps_tuples = self.steps_type.steps()
        self.inf_rows, self.inf_cols = self.steps_type.inf_rows_cols()
        self._wp = None  # warping paths
        if self.use_c:
            self._c_locosettings = loco_cc.LoCoSettings(penalty=self.penalty)

    @staticmethod
    def from_other(lc, series1, series2=None):
        lcn = LocalConcurrences(series1, series2, gamma=lc.gamma, tau=lc.tau, delta=lc.delta,
                                delta_factor=lc.delta_factor, only_triu=lc.only_triu,
                                penalty=lc.penalty, window=lc.window, use_c=lc.use_c)
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
            elif tau_type == 'std':
                diffm = np.std(series)
            elif tau_type == 'mean':
                diffm = np.mean(series)
            else:
                diffm = 1

            if gamma is None:
                # Intuition for gamma:
                # Create an affinity matrix where
                # differences up to the mean/std are in [e^-1, 1],
                # larger differences are i [0, e^-1]
                assert(diffm != 0)
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

    def estimate_settings_from_ssm(self, rho, set_penalty=False,
                                   set_gamma=False, verbose=False):
        """Estimate tau from the self similarity matrix.
        Call before calling the align method.
        Based on Fundamentals of Music Processing.
        """
        if np is None:
            raise AssertionError("The estimate_tau method requires Numpy")
        # TODO: Sampling could be faster than computing the entire matrix
        sdm = np.subtract.outer(self.series1, self.series2)
        if set_gamma:
            # estimate gamma such that in the ssm differences smaller than
            # the mean are in [e^-1, 1], larger differences are in [0, e^-1]
            sdm_mean = np.mean(np.abs(sdm))
            if sdm_mean == 0:
                self.gamma = 1
            else:
                self.gamma = 1 / sdm_mean ** 2
        ssm = np.exp(-self.gamma * np.power(sdm, 2))
        if self.only_triu:
            self.tau = np.quantile(ssm[np.triu_indices(len(ssm))], rho, axis=None)
        else:
            self.tau = np.quantile(ssm, rho, axis=None)
        self.delta = -2 * self.tau
        self.delta_factor = 0.5
        if set_penalty:
            self.penalty = self.tau
        if verbose:
            if self.tau < 1e-5:
                print(f'WARNING: the value of tau ({self.tau}) is very low. '
                      f'Check if the value of gamma ({self.gamma}) is good.')

    def set_penalty_in_ts_domain(self, penalty):
        """Set penalty based on the domain of the time series."""
        self.penalty = 1.0 - np.exp(-self.gamma * penalty ** 2)

    def get_penalty_in_ts_domain(self):
        return np.sqrt(np.log(1.0 + self.penalty)/-self.gamma)

    def align(self):
        """Perform alignment.

        :return: None
        """
        if self._wp is not None:
            return
        _, wp = loco_warping_paths(self.series1, self.series2,
                                   gamma=self.gamma, tau=self.tau, delta=self.delta, delta_factor=self.delta_factor,
                                   only_triu=self.only_triu, penalty=self.penalty,
                                   window=self.window, step_type=self.steps_type,
                                   use_c=self.use_c)
        self._wp = wp
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
        # if self.use_c:
        #     dtw_cc.wps_positivize(self._c_parts, self._wp,
        #                           len(self.series1), len(self.series2),
        #                           0, len(self.series1) + 1,
        #                           0, len(self.series2) + 1,
        #                           False)
        wp = self._wp
        if self.window is None:
            np.abs(wp, out=wp)
            wp[np.isinf(wp)] = -np.inf
        else:
            windowdiff1 = max(0, wp.shape[1] - wp.shape[0])
            windowdiff2 = max(0, wp.shape[0] - wp.shape[1])
            il = np.tril_indices(n=wp.shape[0], k=-1 - self.window - windowdiff2, m=wp.shape[1])
            wp[il] = -np.abs(wp[il])
            il = np.triu_indices(n=wp.shape[0], k=-self.window - windowdiff2, m=wp.shape[1])
            wp[il] = np.abs(wp[il])
            il = np.triu_indices(n=wp.shape[0], k=1 + self.window + windowdiff1, m=wp.shape[1])
            wp[il] = -np.abs(wp[il])
            wp[np.isinf(wp)] = -np.inf
        if self.only_triu:
            il = np.tril_indices(self._wp.shape[0], k=-1)
            wp[il] = -np.inf

    def similarity_matrix(self):
        sm = np.full((len(self.series1) + self.inf_rows,
                      len(self.series2) + self.inf_cols), -inf)
        sm[0:self.inf_rows, 0:self.inf_cols] = 0
        for r in range(len(self.series1)):
            if self.window is None:
                minc, maxc = 0, len(self.series2)
            else:
                minc, maxc = max(0, r - self.window), min(len(self.series2), r + self.window)
            # for c in range(minc):
            #     sm[r + self.inf_rows, c + self.inf_cols] = ma.masked
            for c in range(minc, maxc):
                d = np.exp(-self.gamma * (self.series1[r] - self.series2[c]) ** 2)
                sm[r + self.inf_rows, c + self.inf_cols] = self.delta if d < self.tau else d
            # for c in range(maxc, len(self.series2)):
            #     sm[r + self.inf_rows, c + self.inf_cols] = ma.masked
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
    def wp_data(self):
        return self._wp.data

    def wp_slice_ts(self, rb=None, re=None, cb=None, ce=None,
                    positivize=False, steps_type=None):
        """Slice of the warping paths (based on time series, excluding inf rows/cols).

        :param rb: Begin index in first time series (row index)
        :param re: End index in first time series (row index)
        :param cb: Begin index in second time series (column index)
        :param ce: End index in second time series (column index)
        :param positivize: Make all numbers positive
        :param steps_type: StepsType value
        """
        steps_type = StepsType.wrap(steps_type, steps_type_default)
        inf_cols, inf_rows = steps_type.inf_rows_cols()
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
        # if self.compact:
        #     slice = np.empty((re-rb, ce-cb), dtype=DTYPE)
        #     dtw_cc.wps_expand_slice(self._wp, slice, len(self.series1), len(self.series2),
        #                             rb, re, cb, ce, self._c_dtwsettings)
        cur_slice = self._wp[rb+inf_rows:re+inf_rows,
                             cb+inf_cols:ce+inf_cols]
        if positivize:
            neg_idx = cur_slice < 0
            cur_slice[neg_idx] = -cur_slice[neg_idx]
        return cur_slice

    def best_match(self):
        idx = np.unravel_index(np.argmax(self._wp, axis=None), self._wp.shape)
        r, c = idx
        lcm = LCMatch(self, r, c)
        # path = lcm.path
        # for (x, y) in path:
        #     self._wp[x + 1, y + 1] = ma.masked
        return lcm

    def kbest_matches_store(self, k=1, minlen=2, buffer=0, restart=True, keep=False, matches=None, tqdm=None):
        return self.best_matches_store(k=k, minlen=minlen, buffer=buffer, restart=restart, detectknee=None,
                                       keep=keep, matches=matches, tqdm=tqdm)

    def best_matches_store(self, k=1, minlen=2, buffer=0, restart=True, detectknee=None, keep=False,
                                matches=None, tqdm=None, bufferedargmax=None) -> LCMatches:
        """Get the best matches and store in an LCMatches object.

        :param k:
        :param minlen:
        :param buffer:
        :param restart:
        :param detectknee:
        :param keep: Keep mask to search incrementally for multiple calls of kbest_matches
        :param matches:
        :param tqdm:
        :return:
        """
        import time
        if matches is None:
            matches = LCMatches(self)
        start = time.process_time()
        self.align()
        stop = time.process_time()
        print(f"Computed cumulative cost matrix in {stop - start} seconds")
        it = self.best_matches(k=k, minlen=minlen, buffer=buffer, restart=restart,
                               detectknee=detectknee, bufferedargmax=bufferedargmax)
        if tqdm is not None:
            it = tqdm(it, total=k)
        # tp = time.perf_counter()
        for ki, match in enumerate(it):
            matches.append(match)
            # tn = time.perf_counter()
            # print(f'time: {tn-tp}')
            # tp = tn
        if not keep:
            self._reset_wp_mask()
        return matches

    def best_matches_knee(self, alpha=0.3, thr_value=0, **kwargs):
        return self.best_matches(k=None,
                                 detectknee={'alpha': alpha, 'thr_value': thr_value},
                                 **kwargs)

    def kbest_matches(self, k=1, **kwargs):
        return self.best_matches(k=k, detectknee=None, **kwargs)

    def best_matches(self, k=1, minlen=2, buffer=0, restart=True,
                     detectknee=None, bufferedargmax=None):
        """Yields the next best LocalConcurrent match.
        Stops at k matches (use None for all matches) or when a knee is detected.

        :param k: Number of matches to yield, None is all matches
        :param minlen: Consider only matches of length longer than minlen
        :param buffer: Matches cannot be closer than buffer to each other
        :param restart: Start searching from start, ignore previous calls to kbest_matches
        :param detectknee: Arguments for `util.DetectKnee`
        :param bufferedargmax: Arguments for `util.BufferedArgMax`
        :return: Yield an LCMatch object
        """
        if self._wp is None:
            self.align()
        wp = self._wp
        if restart:
            self._reset_wp_mask()
        if detectknee is not None:
            alpha = detectknee.get('alpha', 0.3)
            alpha_onlyvar = detectknee.get('alpha_onlyvar', alpha/100)
            thr_value = detectknee.get('thr_value', 0)
            dk = util.DetectKnee(alpha=alpha, alpha_onlyvar=alpha_onlyvar,
                                 invert=True, thr_value=thr_value)
            dk.verbose = detectknee.get('verbose', False)
        else:
            dk = None
        if bufferedargmax is None:
            bufferedargmax = {}
        bam = BufferedArgMax(wp, bufferedargmax.get('n', 100))
        l1 = len(self.series1)
        l2 = len(self.series2)
        lperc = max(200, int(l1/10))
        ki = 0
        knee_detected = False
        while (k is None or ki < k) and not knee_detected:
            idx = None
            lcm = None
            wp_value = None
            cnt = 0
            while idx is None and not knee_detected:
                cnt += 1
                if cnt % lperc == 0:
                    print(f'Searching for matches is taking some time (k={ki+1}/{k}: {cnt} tries)')
                bg = bam.get()
                if bg is None:
                    return
                idx = np.unravel_index(bg, wp.shape)
                if idx[0] == 0 or idx[1] == 0:
                    # If all are masked, idx=0 is returned
                    return
                r, c = idx
                # print(f'Best value: wp[{r},{c}] = {wp[r,c]}')
                wp_value = wp[r, c]
                if wp_value < 0:
                    continue
                lcm = LCMatch(self, r, c)
                if detectknee is not None:
                    if dk.dostop(wp_value, only_var=True):
                        knee_detected = True
                path = lcm.path
                for (x, y) in path:
                    x += self.inf_rows
                    y += self.inf_cols
                    # if self.compact:
                    #     changed = dtw_cc.wps_negativize_value(self._c_parts, wp, l1, l2, x, y)
                    #     if not changed:
                    #         # found path contained masked entry, restart
                    #         lcm, idx = None, None
                    #         break
                    if wp[x, y] < 0:
                        # print('found path contains masked, restart')
                        lcm, idx = None, None
                        break
                    else:
                        wp[x, y] = -abs(wp[x, y])

                if len(path) < minlen:
                    # print('found path too short, restart')
                    lcm, idx = None, None
            if buffer < 0 and lcm is not None:
                # Mask the containing rectangle for the path
                miny, maxy = 0, wp.shape[1]
                minx, maxx = 0, wp.shape[0]
                wp[path[0][0]+self.inf_rows:path[-1][0]+self.inf_rows+1, miny:maxy] = -np.abs(wp[path[0][0]+self.inf_rows:path[-1][0]+self.inf_rows+1, miny:maxy])
                wp[minx:maxx, path[0][1]+self.inf_cols:path[-1][1]+self.inf_cols+1] = -np.abs(wp[minx:maxx, path[0][1]+self.inf_cols:path[-1][1]+self.inf_cols+1])
            elif buffer > 0 and lcm is not None:
                # Mask a square around each position in the path
                if self.use_c:
                    loco_cc.loco_path_negativize(path, wp, buffer, self.inf_rows, self.inf_cols)
                else:
                    for p_idx, (x, y) in enumerate(path):
                        # include first row and column with infinities
                        x += self.inf_rows
                        y += self.inf_cols
                        # Reduce buffer towards the extreme points
                        if p_idx < buffer:
                            cbuffer = p_idx + 1
                        elif p_idx > len(path) - buffer:
                            cbuffer = len(path) - p_idx
                        else:
                            cbuffer = buffer
                        # half = int(len(path) / 2)
                        # if p_idx < half:
                        #     cbuffer = math.ceil(buffer * (p_idx + 1) / half)
                        # elif p_idx > len(path) - half:
                        #     cbuffer = math.ceil(buffer * (len(path) - p_idx) / half)
                        # else:
                        #     cbuffer = buffer
                        x_l = max(self.inf_rows, x - cbuffer)
                        x_r = min(x + cbuffer + 1, wp.shape[0])
                        y_l = max(self.inf_cols, y - cbuffer)
                        y_r = min(y + cbuffer + 1, wp.shape[1])
                        wp[x_l:x_r, y_l:y_r] = -np.abs(wp[x_l:x_r, y_l:y_r])
            if lcm is not None:
                ki += 1
                if detectknee is not None:
                    if dk.dostop(wp_value):
                        knee_detected = True
                yield lcm

    def best_path(self, row, col, wp=None):
        """Find the best path starting from the given row and column.

        :param row: Row index in datastructure
        :param col: Column index in datastructure
        :param wp: Warping paths, default is None and taken from the object.
        :return: List of tuples (index in series1, index in series2)
        """
        if self._wp is None:
            return None
        if wp is None:
            wp = self._wp
        l1 = len(self.series1)
        l2 = len(self.series2)
        if self.use_c:
            # print(f'use_c -- {row=}, {col=}, {self.penalty=}')
            p = loco_cc.loco_best_path(wp, l1, l2, row, col, int(l1 / 10),
                                       penalty=self.penalty, step_type=self.steps_type)
            return p
            # pc = p
        # if self.compact:
        #     p = dtw_cc.best_path_compact_affinity(wp, l1, l2, row, col, window=self.window)
        #     return p
        penalties = [self.penalty if sr != sc else 0 for sr, sc in self.steps_tuples]
        i = row
        j = col
        p = []
        while i > 0 and j > 0:
            p.append((i - self.inf_rows, j - self.inf_cols))
            # values = [wp[i - si, j - sj] + penalty for (si, sj), penalty in zip(self.steps_tuples, penalties)]
            # values = [wp[i - 1, j - 1], wp[i, j - 1], wp[i - 1, j]]
            # values = [-1 if v is ma.masked else v for v in values]
            values = []
            for (si, sj), penalty in zip(self.steps_tuples, penalties):
                if wp[i - si, j - sj] >= 0:
                    values.append(wp[i - si, j - sj] + penalty)
                else:
                    values.append(-1)
            c = argmax(values)  # triggers "Warning: converting a masked element to nan"
            if values[c] <= 0:
                break
            best_wp = wp[i - self.steps_tuples[c][0], j - self.steps_tuples[c][1]]
            # if best_wp is ma.masked or best_wp < 0:
            if best_wp < 0:
                assert False
            i -= self.steps_tuples[c][0]
            j -= self.steps_tuples[c][1]
        if p[-1][0] < 0 or p[-1][1] < 0:
            assert False
        p.reverse()
        # print(f"{len(p)=}, {len(pc)=}")
        # print(f"p={p[len(p)-1][0]},{p[len(p)-1][1]} / pc={pc[len(pc)-1, 0]},{pc[len(pc)-1, 1]}")
        # print(f"p={p[0][0]},{p[0][1]} / pc={pc[0, 0]},{pc[0, 1]}")
        # if len(p) != len(pc):
        #     print(f"{wp.shape=}")
        #     loco_cc.save_wp(wp, "/Users/wannes/Desktop/debug/wps.binarray")
        #     print("===p===")
        #     for curr, curc in p:
        #         print(f"[{curr},{curc}]")
        #     print("===pc===")
        #     for curr, curc in pc:
        #         print(f"[{curr},{curc}]")
        #     print(f"Different lengths for {row=} {col=}")
        # for i in range(len(p)):
        #     assert p[i][0] == pc[i, 0]
        #     assert p[i][1] == pc[i, 1]
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

    # def wp_c_print(self):
    #     dtw_cc.wps_print(self._wp, len(self.series1), len(self.series2), window=self.window)

    # def wp_c_print_compact(self):
    #     dtw_cc.wps_print_compact(self._wp, len(self.series1), len(self.series2), window=self.window)


# Based on L. Rabiner and B.-H. Juang. Fundamentals of speech recognition.
# Prentice-Hall, Inc., 1993.
steps_types = {
    "TypeI": ((1, 1), (0, 1), (1, 0)),  # diagonal, go left, go up
    "TypeIII": ((1, 1), (1, 2), (2, 1)),
    "Diagonal": ((1, 1),)
}


class LoCoSettings:
    def __init__(self, only_triu=False,
                       penalty=None, psi=None, window=None,
                       gamma=1, tau=0, delta=0, delta_factor=1,
                       use_c=False, step_type=None):
        self.only_triu = only_triu
        self.penalty = penalty
        self.psi = psi
        self.window = window
        self.gamma = gamma
        self.tau = tau
        self.delta = delta
        self.delta_factor = delta_factor
        self.use_c = use_c
        self.step_type = step_type

    @staticmethod
    def wrap(settings):
        if isinstance(settings, LoCoSettings):
            return settings
        if settings is None:
            return LoCoSettings()
        return LoCoSettings(**settings)

    def kwargs(self):
        return {
            'only_triu': self.only_triu,
            'penalty': self.penalty,
            'psi': self.psi,
            'window': self.window,
            'gamma': self.gamma,
            'tau': self.tau,
            'delta': self.delta,
            'delta_factor': self.delta_factor,
            'use_c': self.use_c,
            'step_type': self.step_type,
        }

    def inf_rows_cols(self):
        if self.step_type == "TypeI":
            inf_rows, inf_cols = 1, 1
        elif self.step_type == "TypeIII":
            inf_rows, inf_cols = 2, 2
        else:
            raise ValueError("Unknown steps type for C version of warping_paths_loco: {}".format(self.step_type))
        return inf_rows, inf_cols

    def steps(self):
        if self.step_type is None:
            return steps_types["TypeI"]
        if self.step_type in steps_types:
            return steps_types[self.step_type]
        raise ValueError("Unknown step_ttype")

    def split_psi(self):
        psi_1b = psi_2b = 0
        if type(self.psi) is int:
            psi_1b = psi_2b = self.psi
        elif type(self.psi) in [tuple, list]:
            psi_1b, psi_2b = self.psi
        return psi_1b, psi_2b

    @property
    def window_value(self):
        if self.window is None:
            return 0
        return self.window


def loco_warping_paths(s1, s2, **kwargs):
    """

    :param step_type: Steps that are agregrated. Tuples with (step in series 1, step in series 2).
        Default is TypeI: (1,1),(1,0),(0,1). A good alternative is TypeIII: (1,2),(2,1),(1,1).
    """
    if np is None:
        raise NumpyException("Numpy is required for the warping_paths method")

    s = LoCoSettings(**kwargs)
    r, c = len(s1), len(s2)

    if s.use_c:
        s1 = util_numpy.verify_np_array(s1)
        s2 = util_numpy.verify_np_array(s2)
        dtw._check_library(raise_exception=True)
        inf_rows, inf_cols = s.inf_rows_cols()
        wps = np.full((r + inf_rows, c + inf_cols), -inf, dtype=DTYPE)
        print(s.kwargs())
        print(f"{wps.shape=} = {wps.shape[0]*wps.shape[1]}")
        loco_cc.loco_warping_paths(wps, s1, s2, ndim=1, **s.kwargs())
        return 0, wps

    steps = s.steps()
    window = s.window_value
    if window == 0:
        window = max(r, c)
    psi_1b, psi_2b = s.split_psi()
    penalties = [s.penalty if sr != sc else 0 for sr, sc in steps]
    steps_rows, steps_cols = zip(*steps)
    inf_rows = max(steps_rows)
    inf_cols = max(steps_cols)
    wps = np.full((r + inf_rows, c + inf_cols), -inf)
    wps[0:inf_rows, 0:psi_2b + inf_cols] = 0
    wps[0:psi_1b + inf_rows, 0:inf_cols] = 0
    for i in range(r):
        j_start = max(0, i - max(0, r - c) - window + 1)
        if s.only_triu:
            j_start = max(i, j_start)
        j_end = min(c, i + max(0, c - r) + window)
        for j in range(j_start, j_end):
            d = np.exp(-s.gamma*(s1[i] - s2[j])**2)
            # print(f"{s1[i] - s2[j]=} -> {d=}")
            wps_prev = max(wps[i + inf_rows - sr, j + inf_cols - sc] - penalty
                           for (sr, sc), penalty in zip(steps, penalties))
            if d < s.tau:
                wps[i + inf_rows, j + inf_rows] = max(0, s.delta + s.delta_factor * wps_prev)
            else:
                wps[i + inf_rows, j + inf_rows] = max(0, d + wps_prev)
    return 0, wps


class BufferedArgMax:
    def __init__(self, a, n=100):
        self.a = a
        self.n = n
        self._idxs = None
        self._yield_idx = None
        self.max_idx = self.a.shape[0] * self.a.shape[1]

    def get(self):
        if self._idxs is None:
            self._populate_idxs()
        rval = self._idxs[self._yield_idx]
        self._yield_idx += 1
        if self._yield_idx == len(self._idxs):
            self._idxs = None
        if rval > self.max_idx:
            return None
        return rval

    def _populate_idxs(self):
        start = time.process_time()
        # ap = self.a.flatten()
        # self._idxs = heapq.nlargest(self.n, range(len(ap)), ap.take)
        # idxs = self.a.flatten().argsort()[-self.n:]
        idxs = np.empty(self.n, dtype=np.int64)
        loco_cc.loco_wps_argmax(self.a, idxs, self.n)
        # idxs = self.a.flatten().argsort(fill_value=0)
        # self._idxs = np.flip(idxs)
        self._idxs = idxs
        self._yield_idx = 0
        stop = time.process_time()
        # print(f"populate idxs: {stop - start} sec")
