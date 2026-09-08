# -*- coding: UTF-8 -*-
"""
dtaidistance.explain.dsw.explainpair
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(requires version 2.5.0 or higher)

Explain the warping path between two time series by finding a piecewise
linear path that is close to the original path. Each segment in the piecewise
linear path has uniform compression/expansion and shift.

Usage:

::

    pair = ExplainPair(ts_a, ts_b, delta_rel=2, delta_abs=0)
    print("DSW distance = ", pair.distance_approx())
    print("DTW distance = ", pair.distance())
    pair.plot_warping("/path/to/dsw.png")


:copyright: Copyright 2025-2026 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import functools
import heapq
import sys
from collections import deque, defaultdict
import dataclasses
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import numpy.typing as npt

from ... import dtw_visualisation as dtwvis
from ... import dtw, affinedtw
from ...dtw import warping_path, DTWSettings, warping_paths
from ...util import SortedList, DDType
from ...preprocessing import scale_linearly, derivative
from . import utils as dsw_utils


diag_angle = np.pi / 4

color_series = "#2E81B9"
color_series_to = "#B92E95"
color_shade = "#F57A00"
color_shade_dark = "#5F3E1C"


class ApproxType(DDType):
    MAX_FACTOR_AND_DIFF_HANDS_ON = "max_factor_and_diff_hands_on"
    MAX_FACTOR_AND_DIFF = "max_factor_and_diff"
    MAX_INDEX = "max_index"
    MEAN_INDEX = "mean_index"  # Not implemented yet
    MAX_FACTOR = "max_factor"
    MAX_DIFF = "max_diff"
    MAX_DIFF_TIMESDIST = "max_diff_timesdist"
    MAX_DIST = "max_dist"
    MAX_FACTOR_LOOSE = "max_factor_loose"
    MAX_FACTOR_AND_DIST = "max_factor_and_dist"


class SplitStrategy(DDType):
    TS_DIFF = "tsdiff"
    SPATIAL_DIST = "spatialdist"
    PATH_DIFF = "pathdiff"  # TODO: depricate
    DERIV = "deriv"
    DERIV_DIST = "derivdist"
    LEAST_TOTAL_COST_DIFF = 'least_total_cost_diff'
    TS_CDIFF = "tscdiff"
    COST_DIFF = "costdiff"
    ONLY_INIT_SPLIT_POINTS = "only_init_split_points"


class FocusOnShape(DDType):
    NONE = "none"
    AFFINE = "affine"
    AFFINE2 = "affine2"
    AFFINEDTW = "affinedtw"
    AFFINEDTW2 = "affinedtw2"
    DERIVATIVE = "derivative"


class PruneStrategy(DDType):
    NONE = "none"  # false
    PRUNE = "prune"  # true
    PRUNE_NOINIT = "prune_noinit"


@dataclass
class Segment:
    s_idx: int
    """Start index in series"""
    e_idx: int
    """End index in series"""
    s_idx_y: int
    """Start index in other series"""
    e_idx_y: int
    """End index in other series"""
    s_idx_p: Optional[int]
    """Start index in path"""
    e_idx_p: Optional[int]
    """End index in path"""
    angle: float  # Angle of path
    """Angle"""
    shift: float
    """Shift left (neg) or right (pos)"""
    elasticity: int  # Difference between dx and dy of path
    """Expansion (pos) or Compression (neg)"""
    ignore: bool = False

    def __post_init__(self):
        if self.ignore:
            self.s_idx_p = None
            self.e_idx_p = None

    def copy(self):
        return dataclasses.replace(self)

    def length(self):
        return self.e_idx - self.s_idx + 1

    @property
    def shift_l(self):
        return 0 if self.shift > 0 else -self.shift

    @property
    def shift_r(self):
        return 0 if self.shift < 0 else self.shift

    @property
    def expansion(self):
        return 0 if self.elasticity < 0 else self.elasticity

    @property
    def compression(self):
        return 0 if self.elasticity > 0 else -self.elasticity

    @property
    def a_expansion(self):
        a = self.angle - diag_angle
        if a < 0:
            return 0
        return a

    @property
    def a_compression(self):
        a = self.angle - diag_angle
        if a > 0:
            return 0
        return -a

    @property
    def ratio_e(self):
        dx = self.e_idx - self.s_idx + 1
        dy = self.elasticity + dx
        if dy < dx:
            return 0
        return dy / dx - 1

    @property
    def ratio_c(self):
        dx = self.e_idx - self.s_idx + 1
        dy = self.elasticity + dx
        if dy > dx:
            return 0
        return dx / dy - 1

    @property
    def ratio(self):
        dx = self.e_idx - self.s_idx + 1
        dy = self.elasticity + dx
        assert dx > 0
        assert dy > 0
        if dy > dx:
            return dy / dx - 1  # expansion
        else:
            return -(dx / dy - 1)  # compression

    @property
    def m_idx(self):
        return (self.s_idx + self.e_idx) / 2

    def compression_logratio(self):
        return np.log(self.e_idx_y - self.s_idx_y + 1) - np.log(self.e_idx - self.s_idx + 1)

    def compression_ratio(self):
        return (self.e_idx_y - self.s_idx_y + 1) / (self.e_idx - self.s_idx + 1)

    def length_y(self):
        return self.e_idx_y - self.s_idx_y + 1

    def s_point(self):
        return (self.s_idx, self.s_idx_y)

    def e_point(self):
        return (self.e_idx, self.e_idx_y)


class ApproxSettings:
    def __init__(
        self,
        approx_type=ApproxType.MAX_FACTOR_AND_DIFF,
        delta_rel=2,
        delta_abs=0.1,
        delta_quantile=None,
        delta_abs_maxlen=None,
        approx_prune=True,
        split_strategy=SplitStrategy.SPATIAL_DIST,
        warp_penalty: Optional[float] = 0.0,
        focus_on_shape=FocusOnShape.NONE,
        init_split_points=None,
        do_remove_singularities=True,
    ):
        """
        Approximation settings for ExplainPair. See more information in the ExplainPair class.
        """
        if not isinstance(approx_type, ApproxType):
            if (type(approx_type) is int or
                    np.issubdtype(type(approx_type), np.integer)):
                approx_type = ApproxType.from_int(approx_type)
            elif type(approx_type) is str:
                approx_type = ApproxType(approx_type)
            else:
                raise ValueError(f'Unknown type for approx_type: {approx_type} ({type(approx_type)})')
        self.approx_type = approx_type
        if delta_abs is None:
            raise ValueError("delta_abs must be provided")
        if self.approx_type == ApproxType.MAX_FACTOR_AND_DIFF and delta_rel is None:
            raise ValueError("delta_rel must be provided for MAX_FACTOR_AND_DIFF")
        self.delta_quantile = delta_quantile
        self.delta_rel = delta_rel
        self.delta_abs = delta_abs
        self.delta_abs_maxlen = delta_abs_maxlen
        self.approx_prune = PruneStrategy.wrap(approx_prune)
        self.warp_penalty = warp_penalty
        if not isinstance(split_strategy, SplitStrategy):
            if (type(split_strategy) is int or
                    np.issubdtype(type(split_strategy), np.integer)):
                split_strategy = SplitStrategy.from_int(split_strategy)
            elif type(split_strategy) is str:
                split_strategy = SplitStrategy(split_strategy)
            else:
                raise ValueError(f'Unknown type for split_strategy: {split_strategy} ({type(split_strategy)})')
        self.split_strategy = split_strategy
        self.focus_on_shape = focus_on_shape
        self.init_split_points = init_split_points
        if self.init_split_points is not None:
            self.init_split_points.sort()
        self.do_remove_singularities = do_remove_singularities

    @staticmethod
    def wrap(s=None, **kwargs):
        if s is None:
            if 'approx_settings' in kwargs:
                s = kwargs['approx_settings']
                if s is None:
                    del kwargs['approx_settings']
                else:
                    assert isinstance(s, ApproxSettings)
                    return s
            return ApproxSettings(**kwargs)
        if isinstance(s, ApproxSettings):
            return s
        if type(s) is dict:
            if 'approx_settings' in s:
                s = s['approx_settings']
                assert isinstance(s, ApproxSettings)
                return s
            return ApproxSettings(**s)
        raise ValueError(f'Unknown argument type for ApproxSettings: {s}')

    def __repr__(self):
        return (
            f"ApproxSettings(approx_type={self.approx_type}, "
            f"delta_rel={self.delta_rel}, delta_abs={self.delta_abs}, "
            f"delta_quantile={self.delta_quantile}, "
            f"delta_abs_maxlen={self.delta_abs_maxlen}, "
            f"approx_prune={self.approx_prune}, "
            f"split_strategy={self.split_strategy}, "
            f"warp_penalty={self.warp_penalty}, "
            f"focus_on_shape={self.focus_on_shape})"
        )

    def kwargs(self):
        return {
            "approx_type": self.approx_type,
            "delta_rel": self.delta_rel,
            "delta_abs": self.delta_abs,
            "delta_quantile": self.delta_quantile,
            "delta_abs_maxlen": self.delta_abs_maxlen,
            "approx_prune": self.approx_prune,
            "split_strategy": self.split_strategy,
            "warp_penalty": self.warp_penalty,
            "focus_on_shape": self.focus_on_shape,
        }

    def c_kwargs(self):
        return {
            "approx_type": self.approx_type.to_int(),
            "delta_rel": self.delta_rel,
            "delta_abs": self.delta_abs,
            "delta_quantile": 0 if self.delta_quantile is None else self.delta_quantile,
            "delta_abs_maxlen": 0 if self.delta_abs_maxlen is None else self.delta_abs_maxlen,
            "approx_prune": self.approx_prune.to_int(),
            "split_strategy": self.split_strategy.to_int(),
            "warp_penalty": 0 if self.warp_penalty is None else self.warp_penalty,
            "focus_on_shape": self.focus_on_shape.to_int(),
        }

    def to_h5_group(self, group):
        for key, value in self.c_kwargs().items():
            group.attrs[key] = value
        if self.init_split_points is not None:
            group.create_dataset('init_split_points', data=np.array(self.init_split_points))


    @staticmethod
    def from_h5_group(group):
        kwargs = {}
        for attr in ["approx_type", "delta_rel", "delta_abs",
                     "delta_quantile", "delta_abs_maxlen",
                     "approx_prune",
                     "split_strategy", "warp_penalty", "focus_on_shape"]:
            if attr in group.attrs:
                kwargs[attr] = group.attrs[attr]
        kwargs["init_split_points"] = (
            group["init_split_points"][:] if "init_split_points" in group else None
        )
        return ApproxSettings(**kwargs)

    @staticmethod
    def estimate_deltaabs_from_noiseamplitude(
        noise_ampl,
        pathlen,
        factor=1.0,
        dtw_settings=None,
    ):
        dtw_settings = DTWSettings.wrap(dtw_settings)
        idcls = dtw_settings.inner_dist_cls()
        delta_abs = idcls.result(factor * pathlen * idcls.inner_val(noise_ampl))
        return delta_abs

    @staticmethod
    def estimate_deltaabs_from_noise(
        *tss,
        pathlen=None,
        factor=1.0,
        window_length=5,
        dtw_settings=None,
    ) -> tuple[float, float]:
        """Estimate delta_abs as the noise on the signal.

        :param tss: List of time series
        :param pathlen: Length of the path, if not given, the length of the
            time series is used
        :param factor: How much noise amplitude to use, number between 0 and 1
        :param window_length: Window length to use for the smoothing
        :param dtw_settings: A DTWSettings (compatible) object
        """
        assert len(tss) > 0
        from scipy.signal import savgol_filter
        dtw_settings = DTWSettings.wrap(dtw_settings)
        if pathlen is None:
            pathlen = len(tss[0])
        noise_ampls = []
        for ts in tss:
            smooth = savgol_filter(ts, window_length=window_length, polyorder=3)
            noise = ts - smooth
            noise_ampl = np.std(noise)
            noise_ampls.append(noise_ampl)
        noise_ampl = float(np.mean(noise_ampls))
        idcls = dtw_settings.inner_dist_cls()
        delta_abs = idcls.result(factor * pathlen * idcls.inner_val(noise_ampl))
        assert isinstance(delta_abs, np.floating) or type(delta_abs) is float
        return float(delta_abs), float(noise_ampl)

    @staticmethod
    def estimate_deltaabs_from_quantiledist(
        s1, s2,
        path=None,
        factor: float = 1.0,
        quantile: float = 0.95,
        dtw_settings: Optional[DTWSettings] = None
    ):
        dtw_settings = DTWSettings.wrap(dtw_settings)
        if path is None:
            path = dtw.warping_path(s1, s2, dtw_settings=dtw_settings)
        dtw_path = np.asarray(path)
        idcls = dtw_settings.inner_dist_cls()
        k = int(np.ceil(len(path) * quantile))
        costs = idcls.inner_dists(s1[dtw_path[:,0]], s2[dtw_path[:,1]])
        indices = np.argpartition(costs, k-1)[:k]
        dtw_dist = idcls.result(np.sum(costs[indices]) / k * len(path))
        assert type(dtw_dist) is float
        delta_abs = factor * dtw_dist
        return delta_abs, dtw_dist

    @staticmethod
    def estimate_deltaabs_from_distmatrix(
        tss,
        delta_abs_quantile=0.95,
        distmatrix=None,
    ):
        """
        Set delta_abs to a quantile of the cluster
        This is similar to the relaxation that is allowed with respect to the
        time series that is the median distance away from the
        medoid of the list of time series.

        :param tss: List (or iterable) of time series
        :param delta_abs_quantile: Use as distance the quantile of all
            distances to the medoid in the list of time series.
        """
        if distmatrix is None:
            distmatrix = dtw.distance_matrix_fast(tss)
        medoid_idx = np.argmin(distmatrix.sum(axis=1))
        delta_abs = np.quantile(distmatrix[medoid_idx, :], delta_abs_quantile)
        return delta_abs, distmatrix

    @staticmethod
    def estimate_penalty_from_histogram(
        tss,
        penalty_aggressiveness: float=0.50,
        distmatrix=None,
    ):
        """
        Set the penalty to the most frequent difference multiplied with
        the aggressiveness when comparing the time series point-to-point
        without warping. This is useful, for example, when the baselines
        differ and this should be allowed.

        :param tss: List (or iterable) of time series
        :param aggressiveness: How aggressive should warping be avoided
            relative to the estimated warp penalty. Lower values mean less
            aggressive, higher value mean more aggressive
        """
        if distmatrix is None:
            distmatrix = dtw.distance_matrix_fast(tss)
        medoid_idx = np.argmin(distmatrix.sum(axis=1))
        tss_prot = tss[medoid_idx]
        penalty = 0
        nb_bins = 20
        for ts in tss:
            hist, bin_edges = np.histogram(np.abs(tss_prot - ts),
                                           bins=nb_bins)
            penalty = max(penalty, bin_edges[np.argmax(hist)])
        penalty *= penalty_aggressiveness
        return penalty, distmatrix


class ExplainPair:
    def __init__(
        self,
        series_from,
        series_to,
        path=None,
        ndim: Optional[int] = None,
        ##
        approx_type=ApproxType.MAX_FACTOR_AND_DIFF,
        delta_rel: float = 1,
        delta_abs: Optional[float] = None,
        delta_abs_maxlen: Optional[int] = None,
        delta_quantile: Optional[float] = None,
        approx_prune: bool = True,
        warp_penalty: Optional[float] = None,
        split_strategy=SplitStrategy.SPATIAL_DIST,
        focus_on_shape=FocusOnShape.NONE,
        init_split_points=None,
        do_remove_singularities=True,
        approx_settings: Optional[ApproxSettings] = None,
        ##
        dtw_settings=None,
        save_intermediates=False,
        variations_on_segments=True,
        get_total_variations=False,
        auto_run=True,
    ):
        """Compute segments and variations that explain the warping path
        between two series by using Dynamic Subsequence Warping.

            Lin, S., Meert, W. Robberechts, P., Blockeel H.,
            "Warping and Matching Subsequences Between Time Series"
            arXiv:2506.15452v1 [cs.LG] 2025

        :param series_from: Series from
        :param series_to: Series to
        :param approx_type: Type of approximation to use.

            Ensures that the new DTW distance after the approximation is within
            a bound. Let d' be the DTW distance of the new path, and d be the
            DTW distance of the original path. The possible choices are:

            * ``max_index``: Absolute position based.
                Allow to deviate from the original path by at most delta_abs
                positions.
                Not related to the d' and d.
                It is the maximum allowed spatial distance between each new
                subpath and the corresponding original subpath.
            * ``max_factor``: Relative distance based.
                :math:`d' \\leq d * (1 + \\delta_{rel})`
            * ``max_factor_loose``: Relative distance based but looser.
                This allows also simplifying subsequences with a very
                low distance. Thus, a good match with a distance close to zero
                and where the simplification would lead to a distance a bit
                higher than zero.

                :math:`d' \\leq d * (1 + 1.1*\\delta_{rel})`
            * ``max_diff``: Absolute distance based:
                :math:`d' \\leq d + \\delta_{abs}`
            * ``max_factor_and_diff``: Combined distance based.
                :math:`d' \\leq d * (1 + \\delta_{rel}) + \\delta_{abs}`
            * ``max_factor_and_diff_hands_on``: Combined distance based, but the input for delta_{abs} is a ratio instead of an absolute value.
                :math:`d' \\leq d * (1 + \\delta_{rel}) + \\delta_{abs} * d`
            * ``max_dist``: Absolute distance based
                :math:`d' <= \\delta_{abs}`
            * ``max_factor_and_dist``: Combined distance based.
                :math:`d' \\leq d * \\delta_{rel} + \\delta_{abs}`

        :param delta_rel: User-defined relative tolerance parameter.
            It controls how much deviation is allowed based on the original
            DTW distance.
            It allows flexibility proportional to the distance of the original
            path.
        :param delta_abs: User-defined absolute tolerance parameter.
            It sets a fixed allowance for deviation.
            It allows flexibility regardless of the distance of the original
            path.
            It has different meanings depending on the approx_type.
        :para delta_abs_maxlen: When applying delta_abs, assume the current
            segment is at most the given length. This reduces trade-off
            effects for long segments between small costs and large costs.
            This is useful if there are long linear segments
            but the distance is different at various places. To avoid that
            a too large distance in one part is compensated by many small
            distances in other parts.
        :param approx_prune: Whether to add a last round that merges segments
            bottom-up.
        :param warp_penalty: Penalty to add when pruning for linear segments
            that are not diagonal. The penalty is added per step that is not
            a diagonal movement. Similar to how a penalty works for DTW. This
            value can be set to (maximally) the difference in y-value that is
            acceptable when the both time series are (approximately)
            horizontal and should thus be warped by one segment.
        :param split_strategy: The strategy to use for deciding the splitting
            points:

            * ``spatialdist``: Split on the point on the path the furthest
                away from the straight path.
            * ``deriv``: Split on the point on the path that has the highest
                local second derivative.
            * ``derivdist``: Split on the point on the path that has the largest
                difference in cost between the point on the path and the
                closest point on the straight path. Approximated with the
                locally computed first and second derivative around the point
                on the path.

        :param path: Use given warping path
        :param dtw_settings: Object of type :class:`DTWSettings`
            This variable is ignored if `path` is given.
        :param save_intermediates: Save intermediate results
        :param init_split_points: Start with these split points and then split
            further. They might be removed by the pruning step.
        :param focus_on_shape: Focus on the shape instead of shape and amplitude.
            One of 'None', 'affine' (scale from series),
            'affine2' (scale to series), 'affinedtw', 'affinedtw2', or
            'derivative'.
        :param variations_on_segments: Compute the variations based on the linear segments instead of the original optimal path
        :param get_total_variations: Whether compute the total variations over all time points on series from. It is useful when we want to know the area regarding the variations on the plotting, but is by default set to be False to save computation.
        """
        self.dtw_settings = DTWSettings.wrap(dtw_settings)
        self.approx_settings = ApproxSettings.wrap(
            approx_type = approx_type,
            delta_rel = delta_rel,
            delta_abs = delta_abs,
            delta_abs_maxlen = delta_abs_maxlen,
            approx_prune = PruneStrategy.wrap(approx_prune),
            split_strategy = SplitStrategy.wrap(split_strategy),
            init_split_points = init_split_points,
            focus_on_shape = FocusOnShape.wrap(focus_on_shape),
            delta_quantile = delta_quantile,
            warp_penalty = warp_penalty,
            do_remove_singularities = do_remove_singularities,
            approx_settings = approx_settings,
        )
        assert self.approx_settings is not None
        settings = self.approx_settings

        self.ndim = 1
        if ndim is not None:
            self.ndim = ndim
        if isinstance(series_from, np.ndarray) and len(series_from.shape) > 1:
            dndim = series_from.shape[1]
            if ndim is None:
                self.ndim = dndim
            else:
                assert self.ndim == dndim
        assert self.ndim is not None and self.dtw_settings.use_ndim == (self.ndim > 1)

        self.series_from, self.series_to = self._transform_series(
            settings.focus_on_shape, series_from, series_to, self.dtw_settings
        )
        if path is not None:
            self.path = path
        else:
            self.path = warping_path(
                self.series_from, self.series_to, **self.dtw_settings.kwargs()
            )

        # Adapt delta_abs
        self.delta_rel = self.approx_settings.delta_rel
        self.delta_abs = self.approx_settings.delta_abs
        if self.delta_abs is None:
            if settings.approx_type == ApproxType.MAX_FACTOR_LOOSE:
                self.delta_abs = 0.1  # will be used to set delta_abs to delta_abs*delta_rel*d
            else:
                self.delta_abs = 0.1
        if settings.approx_type == ApproxType.MAX_FACTOR_AND_DIFF_HANDS_ON:
            assert self.delta_abs is not None and type(self.delta_abs) is float
            if settings.delta_quantile is None:
                self.delta_abs = self.delta_abs * self.dtw_dist()  # update the delta_abs to absolute value for approx_type MAX_FACTOR_AND_DIFF_HANDS_ON
            else:
                self.delta_abs = self.delta_abs * self.dtw_dist_quantile()
                print(f"Setting delta_abs using dtw_dist_quantile: {self.dtw_dist_quantile()} ({self.dtw_dist()}) -> {self.delta_abs=}")
        if self.ndim > 1:
            # When using multivariate data, the delta_rel and delta_abs
            # parameters should be more strict because the deviations tend
            # not to occur all simultaniously.
            self.delta_rel /= np.sqrt(self.ndim)
            self.delta_abs /= np.sqrt(self.ndim)


        self.save_intermediates = save_intermediates
        self.segments: Optional[list[Segment]] = None
        self.line2 = None
        self._segments_notignored: Optional[list[Segment]] = None
        self._variations = None
        self.intermediates = None
        self.init_split_points_idxs = None
        self.variations_on_segments = variations_on_segments

        self.warp_penalty_cost = None
        if settings.warp_penalty is not None:
            _, _, dist2cost = self.dtw_settings.inner_dist_fns()
            self.warp_penalty_cost = dist2cost(settings.warp_penalty)

        self.get_total_variations = get_total_variations

        self._dsw_path = None
        # singularites are not a problem when using ratio_e and ratio_c?
        self.do_remove_singularities = settings.do_remove_singularities
        if auto_run:
            self.compute_segments()

    def __getstate__(self):
        """Prepare state for pickle."""
        state = self.__dict__.copy()
        if 'line_cost' in state:
            # Remove memoized line_cost
            del state['line_cost']
        return state

    def _transform_series(self, focus_on_shape, series_from, series_to, dtw_settings):
        if focus_on_shape == FocusOnShape.NONE:
            return series_from, series_to
        else:
            self.series_from_orig = np.array(series_from)
            self.series_to_orig = np.array(series_to)
        if focus_on_shape == FocusOnShape.AFFINE:
            series_from = scale_linearly(series_from, series_to)
        elif focus_on_shape == FocusOnShape.AFFINE2:
            series_to = scale_linearly(series_to, series_from)
        elif focus_on_shape == FocusOnShape.AFFINEDTW:
            series_from = affinedtw.scale(series_from, series_to, dtw_settings=dtw_settings)
        elif focus_on_shape == FocusOnShape.AFFINEDTW2:
            series_to = affinedtw.scale(series_to, series_from, dtw_settings=dtw_settings)
        elif focus_on_shape == FocusOnShape.DERIVATIVE:
            series_from = derivative(series_from)
            series_to = derivative(series_to)
        return series_from, series_to

    def compute_segments(self):
        self.segments, self.line2, lidxs = self.path_to_segments(self.path)
        self.split_points = [s.s_idx for s in self.segments[1:] if not s.ignore] if len(self.segments) > 1 else []
        self._variations = None  # Amplitude variations (computed lazily)
        if self.get_total_variations:
            self.total_variations = np.sum(self.variations)
        # if (
        #     self.split_strategy == SplitStrategy.ONLY_INIT_SPLIT_POINTS
        #     and self.init_split_points is not None
        # ):
        #     assert np.array_equal(self.init_split_points, self.split_points)

    def segments_notignored(self):
        if self._segments_notignored is None:
            if self.segments is None:
                return None
            self._segments_notignored = [s for s in self.segments if not s.ignore]
        return self._segments_notignored

    def split_points_pathidx(self, include_ends=False):
        if self.segments is None or len(self.segments) <= 1:
            return []
        pidx = set()
        for s in self.segments:
            if s.ignore:
                continue
            pidx.add(s.s_idx_p)
            pidx.add(s.e_idx_p)
        pidx = sorted(pidx)
        if not include_ends:
            pidx = pidx[1:-1]
        return pidx

    def split_points_unique(self):
        """Adjust split points such that there are not two identical numbers
        in the list. The second one is increased by one unless that number is
        in the list, in that case it is dropped. The second to last one is 
        reduced by one if equal to the last one unless the third to last would
        be equal in that case it is dropped.
        """
        sp = self.split_points
        if len(sp) <= 1:
            return sp
        sp2 = [sp[0]]
        for i in range(1, len(sp)-1):
            if sp[i-1] == sp[i]:
                if sp[i] + 1 != sp[i+1]:
                    sp2.append(sp[i] + 1)
                print(f"Same splitpoint {sp[i-1]}=={sp[i]}")
            else:
                sp2.append(sp[i])

        if sp[-2] == sp[-1]:
            if len(sp) > 2 and sp[-3] != sp[-2] - 1:
                sp2[-1] -= 1
                sp2.append(sp[-1])
        else:
            sp2.append(sp[-1])
        return sp2

    def path_to_segments(self, path):
        """Compute segments and variations that explain the warping path between two series.

        :param path: Warping path
        :return: (segments, the simplified path)
        """
        settings = self.approx_settings
        line = np.asarray(path)
        if settings.approx_type == "max_index":
            line2, lidxs = rdp_vectorized(line, epsilon=settings.delta_abs)
        elif settings.split_strategy == SplitStrategy.ONLY_INIT_SPLIT_POINTS:
            line2, lidxs = self.rdp_given(line)
        else:
            line2, lidxs = self.rdp_ssm(line)
        line2 = np.asarray(line2)
        segments = dsw_path_to_segments(line2, lidxs)

        # check for psi-relaxation
        if line[0,0] > 0 or line[0,1] > 0 or line[-1,0] < len(self.series_from) - 1 or line[-1,1] < len(self.series_to) - 1:
            if not self.dtw_settings.use_psi():
                raise Exception("The path does not cover the entire time series (and psi-relaxation is not enabled)")
            segments = self.add_psi_relaxed_segments(line, segments)
        return segments, line2, lidxs

    def add_psi_relaxed_segments(self, line, segments):
        segments2 = []

        if line[0,0] > 0 or line[0,1] > 0:
            ep = line[0]
            dx = ep[0] + 1
            dy = ep[1] + 1
            if dx == 0:
                a = np.pi / 2
            else:
                a = np.arctan(dy / dx)
            shift = ep[1] / 2 - ep[0] / 2
            expansion = dy - dx

            segment = Segment(0, ep[0], 0, ep[1], None, None, a, shift, expansion, True)
            segments2.append(segment)

        segments2.extend(segments)

        if line[-1,0] < len(self.series_from) - 1 or line[-1,1] < len(self.series_to) - 1:
            bp = line[-1]
            ep = [len(self.series_from)-1, len(self.series_to)-1]
            dx = ep[0] - bp[0] + 1
            dy = ep[1] - bp[1] + 1
            if dx == 0:
                a = np.pi / 2
            else:
                a = np.arctan(dy / dx)
            shift = (bp[1] + ep[1]) / 2 - (bp[0] + ep[0]) / 2
            expansion = dy - dx

            segment = Segment(bp[0], ep[0], bp[1], ep[1], None, None, a, shift, expansion, True)
            segments2.append(segment)

        return segments2


    def indices_from_series_to_path(self, points, indices=None):
        settings = self.approx_settings
        if indices is None:
            indices = settings.init_split_points
        assert indices is not None
        if len(indices) == 0:
            return []
        indices = sorted(set(indices))
        assert indices[0] >= points[0][0] and indices[-1] <= points[-1][0], (
            "Path does not cover all given split points"
        )

        # First and last index of series should not be a splitpoint
        if indices[0] == 0:
            indices = indices[1:]
        if len(indices) == 0:
            return []
        if indices[-1] == points[-1,0]:
            indices = indices[:-1]
        if len(indices) == 0:
            return []

        next_split_point_idx = 0
        next_split_point_f = indices[next_split_point_idx]
        add_to_result = []
        for idx in range(len(points)):
            i_f, _ = points[idx]
            stop = False
            while i_f >= next_split_point_f:
                add_to_result.append(idx)
                next_split_point_idx += 1
                if next_split_point_idx >= len(indices):
                    stop = True
                    break
                next_split_point_f = indices[next_split_point_idx]
            if stop:
                break
        return add_to_result

    def _init_queue(self, points):
        init_split_points_o = self.approx_settings.init_split_points
        queue = deque([(0, len(points) - 1)])
        result = set()
        if init_split_points_o is not None and \
            len(init_split_points_o) > 0:
            # Force these split points and continue splitting from this
            # set of splitting points. They can be pruned afterwards.
            init_split_points = init_split_points_o
            if self.dtw_settings.use_psi():
                init_split_points_o.sort()
                if (
                    init_split_points_o[0] < points[0][0]
                    or init_split_points_o[-1] > points[-1][0]
                ):
                    # Retain points on path, ignore others
                    init_split_points = [p for p in init_split_points_o
                                         if points[0][0] <= p <= points[-1][0]]
            add_to_result = self.indices_from_series_to_path(points, init_split_points)
            self.init_split_points_idxs = set(add_to_result)
            if len(add_to_result) == 0:
                return queue, result
            queue = deque([(0, add_to_result[0])])
            for i0, i1 in zip(add_to_result[:-1], add_to_result[1:]):
                queue.append((i0, i1))
            for i0 in add_to_result:
                result.add(i0)
            queue.append((add_to_result[-1], len(points)-1))
            # print(f'Start with queue: {queue} and result: {result}')
        return queue, result

    def get_line_cost_fn(self, inner_dist, include_begin=False, include_end=True):
        """Get a function that computes the linear cost between two points.

        :param inner_dist: Inner distance function
        :param include_begin: Whether to include the begin point
        :param include_end: Whether to include the end point
        :return: Function line_cost(p0, p1)
        """
        @functools.cache
        def line_cost(p0, p1):
            ccosti, _ = self._line_cost(p0, p1, inner_dist,
                                        include_begin=include_begin,
                                        include_end=include_end)
            return ccosti
        return line_cost

    def rdp_setup(self, points):
        # Compute cumulative cost over path
        inner_dist, inner_res, inner_val = self.dtw_settings.inner_dist_fns()
        ccostv_o = np.empty(len(points))  # Cumulative cost vector for original path
        cpenv_o = []
        ccost_o = 0  # Cumulative cost
        prev_diff = 0
        warp_penalty_cost = self.warp_penalty_cost
        for idx in range(len(points)):
            i, j = points[idx]
            cost_o = inner_dist(self.series_from[i], self.series_to[j])
            ccost_o += cost_o
            ccostv_o[idx] = ccost_o
        self.ccostv_o = ccostv_o # Store for later use, but only useful if the savespstrategy in sleeve is pruning. todo: better design to avoid unnecessary storage
        if warp_penalty_cost is not None:
            cpenv_o = np.empty(len(points))  # Cumulative penalty vector for original path
            cpen_o = 0  # Cumulative penalty
            for idx in range(len(points)):
                i, j = points[idx]
                diff = j - i
                if prev_diff != diff:
                    cpen_o += abs(diff - prev_diff) * warp_penalty_cost
                prev_diff = diff
                cpenv_o[idx] = cpen_o
        self.cpenv_o = cpenv_o # Store for later use, but only useful if the savespstrategy in sleeve is pruning. todo: better design to avoid unnecessary storage
        lenr_o = len(points)  # Length of original path remaining

        tolerance_factor_rel, tolerance_factor_ab = (
            self.compute_tolerance_criterion_factors(
                ccost_o, lenr_o, inner_res, inner_val))
        self.tolerance_factor_rel = tolerance_factor_rel
        self.tolerance_factor_ab = tolerance_factor_ab
        # print(f"{tolerance_factor_ab=:_.3f} (with {lenr_o=}) | {tolerance_factor_rel=:_.3f}")


    def rdp_given(self, points):
        assert (
            points[0][0] == 0
            # and points[0][1] == 0
            and points[-1][0] == len(self.series_from) - 1
            # and points[-1][1] == len(self.series_to) - 1
        ), f"Path does not cover full time series: {points[0]}..{points[-1]}"
        self.rdp_setup(points)
        lidxs = self.indices_from_series_to_path(points)
        if len(lidxs) == 0:
            lidxs = [0, len(points) - 1]
        else:
            if lidxs[0] != 0:
                lidxs = [0] + lidxs
            if lidxs[-1] != len(points) - 1:
                lidxs = lidxs + [len(points) - 1]
        line2 = [points[lidx] for lidx in lidxs]
        inner_dist, _, _ = self.dtw_settings.inner_dist_fns()
        self.line_cost = self.get_line_cost_fn(inner_dist)
        return line2, lidxs

    def rdp_ssm(self, points):
        """Simplify the warping path by taking into account the new cost
        according to the Self-Similarity Matrix (SSM).

        This method is based on the Ramer-Douglas-Peucker algorithm to
        simplify a multiline path while retaining its shape.

        Instead of spatial distance to simplifying line, use difference in
        cumulative cost along path. Simplification is allowed based
        on a a criterion that states that the cost of the simplified path
        needs to be smaller than the orginal dtw distance with a certain
        margin.

        The type of criterion and thus the type of relaxation is dependent
        on self.approx_type.

        :param points: warping path. 2D numpy array (points x dimensions)
        """
        split_strategy = self.approx_settings.split_strategy
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        inner_dist, inner_res, inner_val = self.dtw_settings.inner_dist_fns()

        queue, result = self._init_queue(points)
        handled_segments = []

        self.rdp_setup(points)

        # Split selection
        if split_strategy == SplitStrategy.SPATIAL_DIST:
            split_selection = self.max_deviation_from_line
        elif split_strategy == SplitStrategy.PATH_DIFF:
            split_selection = functools.partial(self.max_change_in_path, inner_dist)
        elif split_strategy == SplitStrategy.DERIV:
            from .utils import get_2ndderiv_in_path
            split_selection = functools.partial(
                self.max_2ndderiv_in_path,
                get_2ndderiv_in_path(self.series_from, self.series_to, points),
            )
        elif split_strategy == SplitStrategy.DERIV_DIST:
            from .utils import get_1stderiv_in_path, get_2ndderiv_in_path
            split_selection = functools.partial(
                self.max_deriv_deviation,
                get_1stderiv_in_path(self.series_from, self.series_to, points),
                get_2ndderiv_in_path(self.series_from, self.series_to, points),
            )
        elif split_strategy == SplitStrategy.TS_DIFF:
            split_selection = functools.partial(self.max_ts_diff, inner_dist)
        elif split_strategy == SplitStrategy.TS_CDIFF:
            split_selection = self.max_ts_cdiff
        elif split_strategy == SplitStrategy.COST_DIFF:
            split_selection = functools.partial(self.max_cost_diff, inner_dist)
        else:
            raise AttributeError(f"Unknown split strategy: {split_strategy}")

        line_cost = self.get_line_cost_fn(inner_dist)
        self.line_cost = line_cost

        # Each time for a given segment, either it is accepted, or a new
        # splitting point within the segment is created if the segment is
        # not accepted.
        while len(queue) > 0:
            i0, i1 = queue.popleft()

            if i1 - i0 <= 1:
                # Nothing to simplify
                result.add(i0)
                result.add(i1)
                continue

            p0, p1 = points[i0], points[i1]
            # print(f"== i=[{i0},{i1}] / p=[{p0},{p1}] == td")

            do_simplify = self.do_simplify_segment(
                i0, i1,
                p0, p1,
                self.ccostv_o,
                self.warp_penalty_cost, self.cpenv_o,
                self.tolerance_factor_ab, self.tolerance_factor_rel,
                line_cost,
            )
            if do_simplify:
                # Simplify the path between the current two points
                result.add(i0)
                result.add(i1)
                if self.save_intermediates:
                    handled_segments.append([i0, i1, None])
                # print(f'accepted -- {i0}={points[i0]} {i1}={points[i1]}')
            else:
                # Retry with current largest deviation from the straight
                # line as point in between (like in the original rdp algorithm)
                _, idxmax = split_selection(points, i0, i1)
                if idxmax == i0:
                    idxmax = i0 + 1
                queue.append((i0, idxmax))
                queue.append((idxmax, i1))
                # print(f'split -- {i0}={points[i0]} to {i1}={points[i1]} at {idxmax}={points[idxmax]}')
                if self.save_intermediates:
                    handled_segments.append([i0, i1, idxmax])  # Split the path between the current two points

        if self.approx_settings.approx_prune in [PruneStrategy.PRUNE, PruneStrategy.PRUNE_NOINIT]:
            # print('start pruning')
            result = self.remove_segments(
                points, result,
                self.ccostv_o, self.cpenv_o, line_cost,
                self.tolerance_factor_rel, self.tolerance_factor_ab,
            )

        result = sorted(result)
        if self.save_intermediates:
            self.intermediates = handled_segments
        if self.do_remove_singularities:
            return remove_singularities(points, result)
        return points[result], result

    def remove_segments(self, points, idxs, ccostv_o, cpenv_o, line_cost,
                        tolerance_factor_rel, tolerance_factor_ab):
        """Remove (prune) segments as longs as the cost still satisfies the 
        tolerance criterion.

        Compared to rdp_ssm, which is top-down, this method is bottom-up. 
        Since this is slower, rdp_ssm is used first.

        :param points:  2D numpy array (points x dimensions)
        :param idxs: the idxs to removed, only the retained idxs after the top-down rdp_ssm are passed
        :param ccostv_o: Cumulative cost vector for original path
        :param cpenv_o: Cumulative penalty vector for original path
        :param line_cost: Method to compute linear line cost
        :param tolerance_factor_rel: delta_rel for criterion 
        :param tolerance_factor_abs: delta_abs for criterion
        """
        queue = []
        new_idxs = SortedList(idxs)
        warp_penalty_cost = self.warp_penalty_cost
        approx_prune = self.approx_settings.approx_prune

        for i0, i1, i2 in zip(new_idxs, new_idxs[1:], new_idxs[2:]):
            # The priority is given to the new straight line that would
            # have the lowest cost, basically a weighted length.
            # heapq.heappush(queue, (min(i2 - i1, i1 - i0), (i0, i1, i2)))
            if approx_prune == PruneStrategy.PRUNE_NOINIT and i1 in self.init_split_points_idxs:
                continue
            p0, p2 = points[i0], points[i2]
            c_02a = line_cost((p0[0], p0[1]), (p2[0], p2[1]))
            heapq.heappush(queue, (c_02a, (i0, i1, i2)))

        while len(queue) > 0:
            c_02a, (i0, i1, i2) = heapq.heappop(queue)
            # print(f'== {i0},{i1},{i2} == bu')
            try:
                new_idxs.index(i0)
                new_idxs.index(i1)
                new_idxs.index(i2)
            except ValueError:
                # Already removed
                continue
            p0, p1, p2 = points[i0], points[i1], points[i2]
            # print(p0, p1, p2)
            # cost_old = line_cost((p0[0], p0[1]), (p1[0], p1[1])) + line_cost((p1[0], p1[1]), (p2[0], p2[1]))
            # cost_new = line_cost((p0[0], p0[1]), (p2[0], p2[1]))
            # print(f'Old cost: {cost_old}, new cost: {cost_new}')
            # print(f'diff: {cost_new - cost_old}')
            # self.plot_local_dsw_path(p0, p2, fn=f'local_dsw_{p0}_to_{p2}.png')
            # self.plot_local_dtw_path(p0, p2, fn=f'local_dtw_{p0}_to_{p2}.png')

            do_simplify = self.do_simplify_segment(
                i0, i2,
                p0, p2,
                ccostv_o,
                warp_penalty_cost, cpenv_o,
                tolerance_factor_ab, tolerance_factor_rel,
                line_cost,
            )
            if do_simplify:
                # print(f'Remove {i1}')
                # print(f'merge{points[i0]} to {points[i2]} removing {points[i1]}')
                if approx_prune == PruneStrategy.PRUNE_NOINIT and i0 in self.init_split_points_idxs:
                    pass
                else:
                    try:
                        i_n = new_idxs.find_lt(i0)
                        i0n, i1n, i2n = i_n, i0, i2
                        p0, p2 = points[i0n], points[i2n]
                        c_02a = line_cost((p0[0], p0[1]), (p2[0], p2[1]))
                        heapq.heappush(queue, (c_02a, (i0n, i1n, i2n)))
                        # print(f'Add {i0n}, {i1n}, {i2n}')
                    except ValueError:
                        pass
                if approx_prune == PruneStrategy.PRUNE_NOINIT and i2 in self.init_split_points_idxs:
                    pass
                else:
                    try:
                        i_n = new_idxs.find_gt(i2)
                        i0n, i1n, i2n = i0, i2, i_n
                        p0, p2 = points[i0n], points[i2n]
                        c_02a = line_cost((p0[0], p0[1]), (p2[0], p2[1]))
                        heapq.heappush(queue, (c_02a, (i0n, i1n, i2n)))
                        # print(f'Add {i0n}, {i1n}, {i2n}')
                    except ValueError:
                        pass
                new_idxs.remove(i1)
            else:
                pass

        # print(f'remove_segments: {len(idxs)} -> {cnt[0]=}')
        return new_idxs

    def do_simplify_segment(
        self,
        i0, i1, p0a, p1a,
        ccostv_o,
        warp_penalty_cost, cpenv_o,
        tolerance_factor_ab, tolerance_factor_rel,
        line_cost,
    ):
        """Check whether the current segments meets the simplification criterium.

        :param i0: First index on path
        :param i1: Second index on path
        :param p0a: First point of simplified path
        :param p1a: Second point of simplified path
        """
        ccostp_o = ccostv_o[i1] - ccostv_o[i0]  # Cost of partial original path
        lenp_o = i1 - i0
        if self.approx_settings.delta_abs_maxlen is not None:
            # print(f"lenp_o = max({lenp_o}, {self.delta_abs_maxlen})")
            lenp_o = min(lenp_o, self.approx_settings.delta_abs_maxlen)
        ccostp_a = line_cost((p0a[0], p0a[1]), (p1a[0], p1a[1]))

        if warp_penalty_cost is not None:
            cpenp_a = self.line_warp_penalty(p0a, p1a)
            cpenp_o = cpenv_o[i1] - cpenv_o[i0]
        else:
            cpenp_a = 0
            cpenp_o = 0

        do_simplify = (ccostp_a + cpenp_a) <= (
            max(
                ccostp_o + cpenp_o + lenp_o * tolerance_factor_ab,
                (ccostp_o + cpenp_o) * (1 + tolerance_factor_rel),
            )
        )
        # prec = 4
        # print(f"Part: i=[{i0},{i1}] / p=[{p0a},{p1a}] -> {do_simplify}")
        # print(f"{ccostp_a=:_.3f}+{cpenp_a:.3f} <? "
        #       f"max({ccostp_o=:_.3f} + {lenp_o=}*{tolerance_factor_ab:.3f} + {cpenp_o:.3f}, "
        #     f"{ccostp_o:_.3f}+{ccostp_o:_.3f}*{tolerance_factor_rel:.3f} + {cpenp_o:.3f}*(1+{tolerance_factor_rel:_.3f}))")
        # print(f"{ccostp_a+cpenp_a:_.{prec}f}  <? "
        #       f"max({ccostp_o + cpenp_o + lenp_o * tolerance_factor_ab:_.{prec}f}, "
        #       f"{(ccostp_o + cpenp_o) + (ccostp_o + cpenp_o) * tolerance_factor_rel:_.{prec}f})")
        return do_simplify

    def line_warp_penalty(self, p0, p1):
        """
        Add penalty for every step not on the diagonal.
        """
        cost = (abs(  p1[0] - max(0, p0[0] - p0[1])
                    - p1[1] + max(0, p0[1] - p0[0]))
                * self.warp_penalty_cost)
        return cost

    def compute_tolerance_criterion_factors(self, ccost, length, cost2dist, dist2cost):
        """Compute the tolerance criterion factors that are allowed for the approximation of the
        current segment.

        Note: unlike the paper, where the absolute tolerance criterion doesn't consider the total path length (L),
        our implementation incorporates L from the start, since it's known in advance. This avoids repeated divisions.

        :param ccost: Cumulative cost of total path
        :param length: length of total path
        :param cost2dist: Transform cost to distance, corresponding to phi^{-1} in the paper.
        :param dist2cost: Transform distance to cost, corresponding to phi in the paper.
        :returns: (relative tolerance criterion factor, absolute tolerance criterion factor)
        """
        # ccost = cost2dist(ccost)
        approx_type = self.approx_settings.approx_type
        delta_rel = self.delta_rel
        delta_abs = self.delta_abs
        if approx_type == ApproxType.MAX_FACTOR:
            ccost_ub = cost2dist(ccost) * delta_rel
            try:
                ub_m = dist2cost(ccost_ub) / ccost
            except (ValueError, ZeroDivisionError):
                ub_m = 0
            ub_a = 0
        elif approx_type == ApproxType.MAX_FACTOR_LOOSE:
            assert delta_rel is not None
            assert delta_abs is not None
            ccost_ub = cost2dist(ccost) * (delta_rel)
            try:
                ub_m = dist2cost(ccost_ub) / ccost
            except (ValueError, ZeroDivisionError):
                ub_m = 0
            ub_a = (dist2cost(cost2dist(ccost)*(1 + delta_rel*delta_abs)) - ccost) / length
        elif approx_type in [ApproxType.MAX_FACTOR_AND_DIFF, ApproxType.MAX_FACTOR_AND_DIFF_HANDS_ON]:
            ccost_ub = cost2dist(ccost) * delta_rel
            if ccost == 0.0:
                ub_m = 0
            else:
                try:
                    ub_m = dist2cost(ccost_ub) / ccost
                except (ValueError, ZeroDivisionError):
                    ub_m = 0
            ub_a = (dist2cost(cost2dist(ccost) + delta_abs) - ccost) / length
        elif approx_type == ApproxType.MAX_DIFF:
            ub_m = 0
            ub_a = (dist2cost(cost2dist(ccost) + delta_abs) - ccost) / length
        elif approx_type == ApproxType.MAX_DIFF_TIMESDIST:
            assert delta_abs is not None
            ub_m = 0
            ub_a = (dist2cost(cost2dist(ccost) * (1 + delta_abs)) - ccost) / length
        elif approx_type == ApproxType.MAX_DIST or approx_type == ApproxType.MAX_INDEX:
            ub_m = 0
            ub_a = (dist2cost(delta_abs) - ccost) / length
        elif approx_type == ApproxType.MAX_FACTOR_AND_DIST:
            if ccost == 0.0:
                ub_m = 0
            else:
                try:
                    ub_m = dist2cost(cost2dist(ccost) * delta_rel) / ccost
                except (ValueError, ZeroDivisionError):
                    ub_m = 0
            ub_a = (dist2cost(delta_abs) - ccost) / length
        else:
            raise ValueError(f'Unknown approximation type: {approx_type}')

        # print(f'{ub_m=:.4f}, {ub_a=:.4f}, {ccosti=:.4f}, {ccostie=:.4f}~{inner_res(ccostie):.4f}, {ccostia=:.4f}~{inner_res(ccostia):.4f}, {length=}')
        return ub_m, ub_a

    def _max_deviation_from_line_filtered(self, points, i0, i1, idx_filter,
                                          use_spatial=True, inner_dist=None):
        """Find maximal deviation from line [i0,i1], but only use
        points available in idx_filter.
        """
        p0, p1 = points[i0], points[i1]
        p0p1norm = np.linalg.norm(p1 - p0)
        p0p1normsqr = p0p1norm ** 2
        idxmax = i0
        distmax = 0
        i_i0 = -1
        for i_res, idx in enumerate(idx_filter):
            if idx < i0 or idx > i1:
                continue
            if idx == i0:
                i_i0 = i_res
                if idx_filter[i_res + 1] == i1:
                    return 0, i0, i_i0
            p = points[idx]
            if np.allclose(p0, p1):
                if use_spatial:
                    dist = np.linalg.norm(p - p0)
                else:
                    dist, _ = self._line_cost(p, p0, inner_dist)
            else:
                t = ((p[0] - p0[0]) * (p1[0] - p0[0]) + (p[1] - p0[1]) * (p1[1] - p0[1])
                     ) / p0p1normsqr
                if t < 0:
                    if use_spatial:
                        dist = np.linalg.norm(p - p0)
                    else:
                        dist, _ = self._line_cost_alldir(p, p0, inner_dist)
                elif t > 1:
                    if use_spatial:
                        dist = np.linalg.norm(p - p1)
                    else:
                        dist, _ = self._line_cost_alldir(p, p1, inner_dist)
                else:
                    pt = np.array([int(p0[0] + t * (p1[0] - p0[0])),
                                   int(p0[1] + t * (p1[1] - p0[1]))])
                    if use_spatial:
                        dist = np.linalg.norm(p - pt)
                    else:
                        dist, _ = self._line_cost_alldir(p, pt, inner_dist)
                        # print(f'{idx=}, {dist=:.4f}, {p=}, {pt=}, sd={np.linalg.norm(p - pt):.4f}')
            if dist > distmax:
                distmax = dist
                idxmax = idx
        return distmax, idxmax, i_i0

    def max_change_in_path(self, inner_dist, points, i0, i1, only_from=False):
        # TODO: we probably can save on inner_cost calculations by storing the costs over the path
        # Project vert and hor. The idea is that this is to the points that represent
        # the situation when there is no warping (at the start, this is the diagonal).
        # Both vert and hor is to be symmetric if s1 and s2 switch.

        p0, p1 = points[i0], points[i1]
        # The warping path is always concacve wrt the linear interpolated path.
        # Thus, the vertical and horizontal projections always end up on the linear path.
        s_tf = (p1[1] - p0[1]) / (p1[0] - p0[0])  # slope for vertical projection
        s_ft = (p1[0] - p0[0]) / (p1[1] - p0[1])  # slope for horizontal projection
        i_f = p0[0]
        i_t = p0[1]
        max_i_lf = len(self.series_from) - 1
        max_i_lt = len(self.series_to) - 1

        diff_max = 0
        idx_max = i0
        for idx in range(i0 + 1, i1):
            i_of, i_ot = points[idx]
            c_o = inner_dist(self.series_from[i_of], self.series_to[i_ot])
            # Horizontal (project along to-axis, to the from-series)
            i_lf = min(int(s_tf * (i_ot - i_t) + i_f), max_i_lf)  # not identical to bresenham's
            c_l = inner_dist(self.series_from[i_lf], self.series_to[i_ot])
            diff = c_l - c_o
            if not only_from:
                # Vertical (project along from-axis, to the to-series)
                i_lt = min(int(s_ft * (i_of - i_f) + i_t), max_i_lt)  # not identical to bresenham's
                c_l = inner_dist(self.series_from[i_of], self.series_to[i_lt])
                diff = max(diff, c_l - c_o)
            if diff > diff_max:
                diff_max = diff
                idx_max = idx
        return diff_max, idx_max

    def max_2ndderiv_in_path(self, ders, points, i0, i1, only_from=False):
        """Find the point in the path that has the highest second
        derivative."""
        der_max = 0
        idx_max = i0
        for idx in range(i0 + 1, i1):
            der = ders[idx]
            if der > der_max:
                der_max = der
                idx_max = idx
        return der_max, idx_max

    def max_deriv_deviation(self, ders1, ders2, points, i0, i1):
        """Find the point in the path that is a balance between the
        highest 2nd derivative and the furthest away from the line
        between the two indices. This is computed using the second-order
        Taylor expansion to approximate the difference in the cost matrix

        :param ders1: First derivative of each point
        :param ders2: Second derivative of each point
        :param points: List of all points
        :param i0: Start index in points
        :param i1: End index in points
        :return: Value, Index
        """
        # Dists close to zero have to much variation for small changes.
        # For example dist=1 and dist=2 double in distance making that
        # dist=2 is often preferred and the derivatives are ignored.
        # Soften this effect (e.g. like a additive smoothing)
        # Higher values prefer the derivative, lower the distance.
        min_dist = 10
        p0, p1 = points[i0], points[i1]
        p0p1norm = np.linalg.norm(p1 - p0)
        p0p1normsqr = p0p1norm ** 2
        distmax = 0
        idxmax = i0
        for idx in range(i0, i1):
            p = points[idx]
            if np.allclose(p0, p1):
                dist = np.linalg.norm(p - p0)
            else:
                # Perpendicular distance
                # a = np.linalg.norm(np.cross(p - p0, p0p1diff))
                # assert a >= 0
                # dist = np.divide(a, p0p1norm)
                # Closest distance (point might be beyond line segment
                t = ((p[0] - p0[0]) * (p1[0] - p0[0]) +
                     (p[1] - p0[1]) * (p1[1] - p0[1])) / p0p1normsqr
                if t < 0:
                    dist = np.linalg.norm(p - p0)
                elif t > 1:
                    dist = np.linalg.norm(p - p1)
                else:
                    pt = np.array([p0[0] + t * (p1[0] - p0[0]),
                                   p0[1] + t * (p1[1] - p0[1])])
                    dist = np.linalg.norm(p - pt)
            # First- or second-order Taylor expansion to approximate the
            # difference in the self-similarity matrix (with abs cost 
            # function) based on the local derivatives
            dist += min_dist
            der1 = ders1[idx]
            if ders2 is None:
                dist2 = der1*dist
            else:
                der2 = ders2[idx]
                dist2 = der1*dist + 1/2*der2*dist**2
            # print(f"{idx}: {dist2:.3f} = {der1:.3f}*{dist:.3f} + 1/2*{der2:.3f}*{dist**2:.3f}")
            dist = dist2

            if dist > distmax:
                distmax = dist
                idxmax = idx
        return distmax, idxmax

    def max_ts_diff(self, innerdist, points, i0, i1):
        """Strategy for splitting (tsdiff).

        Assuming that when this paths is preplaced by a straight line
        segment, each point on the path is replaced by the closests point
        on the straight line path. The cost of the replacement is the sum
        of the differences on the time series between the two points.
        """
        p0, p1 = points[i0], points[i1]
        p0p1norm = np.linalg.norm(p1 - p0)
        p0p1normsqr = p0p1norm ** 2
        diffmax = 0
        idxmax = i0
        for idx in range(i0, i1):
            p = points[idx]
            # Find point on straight line, closest to current point
            if np.allclose(p0, p1):
                pt = p0
            else:
                # Closest distance (point might be beyond line segment
                t = ((p[0] - p0[0]) * (p1[0] - p0[0]) +
                     (p[1] - p0[1]) * (p1[1] - p0[1])) / p0p1normsqr
                if t < 0:
                    pt = p0
                elif t > 1:
                    pt = p1
                else:
                    pt = np.array([int(p0[0] + t * (p1[0] - p0[0])),
                                   int(p0[1] + t * (p1[1] - p0[1]))])
            diff = (np.sum(np.abs(self.series_from[p[0]] - self.series_from[pt[0]])) +
                    np.sum(np.abs(self.series_to[p[1]] - self.series_to[pt[1]])))
            if diff > diffmax:
                diffmax = diff
                idxmax = idx
        return diffmax, idxmax

    def all_ts_diff(self, points, i0, i1):
        """Return all values for the tsdiff strategy for splitting.
        Meant for visualization.
        """
        p0, p1 = points[i0], points[i1]
        p0p1norm = np.linalg.norm(p1 - p0)
        p0p1normsqr = p0p1norm ** 2
        diffs = np.zeros((i1-i0+1))
        for i_d, idx in enumerate(range(i0, i1+1)):
            p = points[idx]
            # Find point on straight line, closest to current point
            if np.allclose(p0, p1):
                pt = p0
            else:
                # Closest distance (point might be beyond line segment
                t = ((p[0] - p0[0]) * (p1[0] - p0[0]) +
                     (p[1] - p0[1]) * (p1[1] - p0[1])) / p0p1normsqr
                if t < 0:
                    pt = p0
                elif t > 1:
                    pt = p1
                else:
                    pt = np.array([int(p0[0] + t * (p1[0] - p0[0])),
                                   int(p0[1] + t * (p1[1] - p0[1]))])
            # 
            diff = (abs(self.series_from[p[0]] - self.series_from[pt[0]]) +
                    abs(self.series_to[p[1]] - self.series_to[pt[1]]))  # difference that would be added to current cost
            diffs[i_d] = diff
        return diffs

    def max_ts_cdiff(self, points, i0, i1):
        p0, p1 = points[i0], points[i1]
        p0p1norm = np.linalg.norm(p1 - p0)
        p0p1normsqr = p0p1norm ** 2
        diffmax = 0
        idxmax = i0
        for idx in range(i0, i1):
            p = points[idx]
            if np.allclose(p0, p1):
                pt = p0
            else:
                # Closest distance (point might be beyond line segment
                t = ((p[0] - p0[0]) * (p1[0] - p0[0]) +
                     (p[1] - p0[1]) * (p1[1] - p0[1])) / p0p1normsqr
                if t < 0:
                    pt = p0
                elif t > 1:
                    pt = p1
                else:
                    pt = np.array([int(p0[0] + t * (p1[0] - p0[0])),
                                   int(p0[1] + t * (p1[1] - p0[1]))])
            # diff = (np.abs(np.sum(self.series_from[p[0]] - self.series_from[p[0]:pt[0]+1])) +
            #         np.abs(np.sum(self.series_to[p[1]] - self.series_to[p[1]:pt[1]+1])))
            diff = (abs(pt[0]-p[0])*(abs(self.series_from[p[0]] - self.series_from[pt[0]])) +
                    abs(pt[1]-p[1])*(abs(self.series_to[p[1]] - self.series_to[pt[1]])))
            if diff > diffmax:
                diffmax = diff
                idxmax = idx
        return diffmax, idxmax

    def max_cost_diff(self, innerdist, points, i0, i1):
        p0, p1 = points[i0], points[i1]
        p0p1norm = np.linalg.norm(p1 - p0)
        diffmax = 0
        idxmax = i0
        rc = (p1[1]-p0[1]) / (p1[0] - p0[0])
        for idx in range(i0, i1):
            p = points[idx]
            pa1 = [p[0], min(len(self.series_to)-1,int(rc*(p[0]-p0[0]+p0[1])))]
            pa0 = [max(0,int(1/rc*(p[1]-p0[1])+p0[0])), p[1]]
            c = innerdist(self.series_from[p[0]], self.series_to[p[1]])
            ca = (innerdist(self.series_from[pa0[0]], self.series_to[pa0[1]])
                  + innerdist(self.series_from[pa1[0]], self.series_to[pa1[1]])) / 2
            diff = ca - c
            if diff > diffmax:
                diffmax = diff
                idxmax = idx
        return diffmax, idxmax

    def max_deviation_from_line(self, points, i0, i1):
        """Find the point in the path that is the furthest away from the line between the two indices."""
        p0, p1 = points[i0], points[i1]
        p0p1norm = np.linalg.norm(p1 - p0)
        p0p1normsqr = p0p1norm ** 2
        distmax = 0
        idxmax = i0
        for idx in range(i0, i1):
            p = points[idx]
            if np.allclose(p0, p1):
                dist = np.linalg.norm(p - p0)
            else:
                # Perpendicular distance
                # a = np.linalg.norm(np.cross(p - p0, p0p1diff))
                # assert a >= 0
                # dist = np.divide(a, p0p1norm)
                # Closest distance (point might be beyond line segment
                t = (
                            (p[0] - p0[0]) * (p1[0] - p0[0]) + (p[1] - p0[1]) * (p1[1] - p0[1])
                    ) / p0p1normsqr
                if t < 0:
                    dist = np.linalg.norm(p - p0)
                elif t > 1:
                    dist = np.linalg.norm(p - p1)
                else:
                    pt = np.array(
                        [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])]
                    )
                    dist = np.linalg.norm(p - pt)
            if dist > distmax:
                distmax = dist
                idxmax = idx
        return distmax, idxmax

    def distance(self,with_warp_penalty=False) -> float:
        dist, _ = self.distance_per_segment(with_warp_penalty=with_warp_penalty)
        return dist

    def distance_per_segment(self, with_warp_penalty=False):
        if self.segments is None:
            raise Exception("No segments defined")
        dist = 0
        inner_dist, inner_res, _ = self.dtw_settings.inner_dist_fns()
        warp_penalty_cost = self.warp_penalty_cost
        prev_diff = 0
        dists = []
        for segment in self.segments:
            cost_seg = 0
            if segment.ignore:
                point = segment.e_point()
                idist = inner_dist(self.series_from[point[0]], self.series_to[point[1]])
            else:
                assert segment.s_idx_p is not None and segment.e_idx_p is not None
                for i_p in range(segment.s_idx_p, segment.e_idx_p):
                    point = self.path[i_p]
                    cost_seg += inner_dist(self.series_from[point[0]], self.series_to[point[1]])
                    if with_warp_penalty and warp_penalty_cost is not None:
                        diff = point[1] - point[0]
                        if prev_diff != diff:
                            cost_seg += abs(diff - prev_diff) * warp_penalty_cost
                        prev_diff = diff
            dist += cost_seg
            dists.append(cost_seg)
        segment = self.segments[-1]
        if segment.ignore:
            idist = 0
        else:
            point = segment.e_point()
            idist = inner_dist(self.series_from[point[0]], self.series_to[point[1]])
        dists.append(idist)
        dist += idist
        dist = inner_res(dist)
        assert type(dist) is float
        return dist, dists

    def distance_approx(self, with_warp_penalty=False) -> float:
        dist, _ = self.distance_approx_per_segment(with_warp_penalty=with_warp_penalty)
        return dist

    def distance_approx_per_segment(self, with_warp_penalty=False):
        """DTW Distance for approximated path."""
        if self.segments is None:
            raise Exception("No segments available")
        dist = 0
        inner_dist, inner_res, _inner_val = self.dtw_settings.inner_dist_fns()
        dists = []

        for segment in self.segments:
            if segment.ignore:
                cost_seg = 0
            else:
                p0 = (segment.s_idx, segment.s_idx_y)
                p1 = (segment.e_idx, segment.e_idx_y)
                cost_seg, _ = self._line_cost(p0, p1, inner_dist,
                                            include_begin=True,
                                            include_end=False)

                if with_warp_penalty:
                    cost_seg += self.line_warp_penalty(p0, p1)
            dist += cost_seg
            dists.append(cost_seg)
        segment = self.segments[-1]
        i_f, i_t = segment.e_idx, segment.e_idx_y
        if segment.ignore:
            cost_seg = 0
        else:
            cost_seg = inner_dist(self.series_from[i_f], self.series_to[i_t])
        dists.append(cost_seg)
        dist += cost_seg
        dist = inner_res(dist)
        assert type(dist) is float
        return dist, dists

    def from_indices(self):
        assert self.segments is not None
        idxs = [s.s_idx for s in self.segments]
        idxs.append(self.segments[-1].e_idx)
        return idxs

    def dsw_path(self):
        """The piece-wise linearized full path."""
        if self._dsw_path is None:
            self._dsw_path = self.segments_to_path()
        return self._dsw_path

    def dsw_dist(self):
        """The distance along the approximated (DSW) path."""
        dsw_path = np.array(self.dsw_path())
        idcls = self.dtw_settings.inner_dist_cls()
        dsw_dist = idcls.result(np.sum(idcls.inner_dists(self.series_from[dsw_path[:,0]], self.series_to[dsw_path[:,1]])))
        return dsw_dist

    def dtw_dist(self) -> float:
        """The distance along the original DTW path."""
        dtw_path = np.array(self.path)
        idcls = self.dtw_settings.inner_dist_cls()
        dtw_dist = idcls.result(np.sum(idcls.inner_dists(self.series_from[dtw_path[:,0]], self.series_to[dtw_path[:,1]])))
        assert type(dtw_dist) is float
        return dtw_dist

    def dtw_dist_quantile(self):
        """The distance along the original DTW path but ignoring the highest inner costs."""
        assert self.approx_settings.delta_quantile is not None
        _, dtw_dist = ApproxSettings.estimate_deltaabs_from_quantiledist(
            self.series_from, self.series_to, self.path,
            factor=1.0,
            quantile=self.approx_settings.delta_quantile,
            dtw_settings=self.dtw_settings,
        )
        assert type(dtw_dist) is float
        return dtw_dist

    def check_is_local_tolerance_satisfied(self, verbose=False):
        assert hasattr(self, "ccostv_o") and self.ccostv_o is not None
        assert hasattr(self, "cpenv_o") and self.cpenv_o is not None
        inner_dist, _, _ = self.dtw_settings.inner_dist_fns()
        line_cost = self.get_line_cost_fn(inner_dist)
        satisfies_all = True
        assert self.segments is not None
        cnt_satis = 0
        satisfied = np.zeros(len(self.segments), dtype=np.bool_)
        for segi, segment in enumerate(self.segments):
            if segment.ignore:
                satisfies = True
            else:
                satisfies = self.do_simplify_segment(
                    segment.s_idx_p,
                    segment.e_idx_p,
                    segment.s_point(),
                    segment.e_point(),
                    self.ccostv_o,
                    self.warp_penalty_cost,
                    self.cpenv_o,
                    self.tolerance_factor_ab,
                    self.tolerance_factor_rel,
                    line_cost,
                )
            if satisfies:
                cnt_satis += 1
                satisfied[segi] = True
            elif verbose:
                print(f"Crit fails: {segment}")
            satisfies_all = satisfies_all and satisfies
        if verbose:
            print(f"Local check satisfied: {satisfies_all} ({cnt_satis}/{len(self.segments)})")
        return satisfies_all, satisfied

    def check_is_global_tolerance_satisfied(self):
        dtw_dist =  self.dtw_dist()
        dsw_dist = self.dsw_dist()
        assert type(dsw_dist) is float
        approx_type = self.approx_settings.approx_type
        delta_rel = self.delta_rel
        delta_abs = self.delta_abs
        assert type(delta_rel) is float and type(delta_abs) is float

        if approx_type in [ApproxType.MAX_FACTOR_AND_DIFF, ApproxType.MAX_FACTOR_AND_DIFF_HANDS_ON]:
            global_bound =  (1 + delta_rel) * dtw_dist + delta_abs
        elif approx_type == ApproxType.MAX_FACTOR_LOOSE:
            global_bound = (1 + (1 + delta_abs) * delta_rel) * dtw_dist
        elif approx_type == ApproxType.MAX_FACTOR:
            global_bound = (1 + delta_rel) * dtw_dist
        elif approx_type == ApproxType.MAX_DIFF:
            global_bound =  dtw_dist + delta_abs
        elif approx_type == ApproxType.MAX_DIFF_TIMESDIST:
            global_bound = dtw_dist * (1 + delta_abs)
        elif approx_type == ApproxType.MAX_DIST:
            global_bound =  delta_abs
        elif approx_type == ApproxType.MAX_FACTOR_AND_DIST:
            global_bound =  delta_rel * dtw_dist + delta_abs
        elif approx_type == ApproxType.MAX_INDEX:
            print("Warning: MAX_INDEX does not consider distance difference.")
            return True
        else:
            raise ValueError(f'Unknown approximation type: {approx_type}')
        assert type(global_bound) is float
        satisfied = dsw_dist <= global_bound
        print(f"satisfied: {satisfied}, approx dist: {dsw_dist:.2f}, dtw dist: {dtw_dist:.2f}, bound: {global_bound:.2f}")
        return satisfied

    def segments_to_path(self, segments=None):
        """Approximated (linearized) full path."""
        if segments is None:
            segments = self.segments
        if segments is None:
            raise Exception("No segments found")
        path = []
        for segment in segments:
            if segment.ignore:
                path.append((segment.s_idx, segment.s_idx_y))
                continue
            d_f = segment.e_idx - segment.s_idx
            d_t = - (segment.e_idx_y - segment.s_idx_y)
            error = d_f + d_t
            i_f, i_fe = segment.s_idx, segment.e_idx
            i_t, i_te = segment.s_idx_y, segment.e_idx_y

            path_segment = []
            while True:
                if i_f == i_fe and i_t == i_te:
                    break
                path_segment.append((i_f, i_t))
                e2 = 2 * error
                if e2 >= d_t:
                    error += d_t
                    i_f += 1
                if e2 <= d_f:
                    error += d_f
                    i_t += 1
            path.extend(path_segment)
        if not segments[-1].ignore:
            path.append((segments[-1].e_idx, segments[-1].e_idx_y))
        return path

    def _line_cost(self, p0, p1, inner_dist, include_begin=True, include_end=True):
        # Bresenham's line algorithm
        d_f = p1[0] - p0[0]
        d_t = - (p1[1] - p0[1])
        error = d_f + d_t
        i_f, i_fe = p0[0], p1[0]
        i_t, i_te = p0[1], p1[1]
        ccosti_n = 0
        approx_len = 0
        while True:
            ccosti_n += inner_dist(self.series_from[i_f], self.series_to[i_t])
            approx_len += 1
            if i_f == i_fe and i_t == i_te:
                break
            e2 = 2 * error
            if e2 >= d_t:
                error += d_t
                i_f += 1
            if e2 <= d_f:
                error += d_f
                i_t += 1
        if not include_begin:
            approx_len -= 1
            ccosti_n -= inner_dist(self.series_from[p0[0]], self.series_to[p0[1]])
        if not include_end:
            approx_len -= 1
            ccosti_n -= inner_dist(self.series_from[i_fe], self.series_to[i_te])
        return ccosti_n, approx_len

    def _line_cost_with_path(self, p0, p1, inner_dist, include_begin=True, include_end=True):
        # Bresenham's line algorithm
        d_f = p1[0] - p0[0]
        d_t = - (p1[1] - p0[1])
        error = d_f + d_t
        i_f, i_fe = p0[0], p1[0]
        i_t, i_te = p0[1], p1[1]
        ccosti_n = 0
        approx_len = 0
        path = []
        while True:
            ccosti_n += inner_dist(self.series_from[i_f], self.series_to[i_t])
            path.append((i_f, i_t))
            approx_len += 1
            if i_f == i_fe and i_t == i_te:
                break
            e2 = 2 * error
            if e2 >= d_t:
                error += d_t
                i_f += 1
            if e2 <= d_f:
                error += d_f
                i_t += 1
        if not include_begin:
            approx_len -= 1
            ccosti_n -= inner_dist(self.series_from[p0[0]], self.series_to[p0[1]])
            path.remove((p0[0], p0[1]))
        if not include_end:
            approx_len -= 1
            ccosti_n -= inner_dist(self.series_from[i_fe], self.series_to[i_te])
            path.remove((i_fe, i_te))
        return ccosti_n, approx_len, path

    def _line_cost_alldir(self, p0, p1, inner_dist, include_begin=True, include_end=True):
        """Bresenham's line algorithm for all possible directions, not only upwards.
        """
        d_f = abs(p1[0] - p0[0])
        s_f = 1 if p0[0] < p1[0] else -1
        d_t = -abs(p1[1] - p0[1])
        s_t = 1 if p0[1] < p1[1] else -1
        error = d_f + d_t
        i_f, i_fe = p0[0], p1[0]
        i_t, i_te = p0[1], p1[1]
        ccosti_n = 0
        approx_len = 0

        while True:
            ccosti_n += inner_dist(self.series_from[i_f], self.series_to[i_t])
            approx_len += 1
            if i_f == i_fe and i_t == i_te:
                break
            e2 = 2 * error
            if e2 >= d_t:
                error += d_t
                i_f += s_f
            if e2 <= d_f:
                error += d_f
                i_t += s_t
        if not include_begin:
            approx_len -= 1
            ccosti_n -= inner_dist(self.series_from[p0[0]], self.series_to[p0[1]])
        if not include_end:
            approx_len -= 1
            ccosti_n -= inner_dist(self.series_from[i_fe], self.series_to[i_te])
        return ccosti_n, approx_len

    @property
    def variations(self):
        if self._variations is not None:
            return self._variations
        variations = self.get_variations(on_segments = self.variations_on_segments)
        self._variations = variations
        return self._variations

    def get_variations(self, on_segments=False, amplitude_on_series_from=True):
        """Compute the amplitude variations

        :param on_segments: Compute the variations based on the linear segments
            instead of the original optimal path
        :param amplitude_on_series_from: Whether the amplitude variations are computed with respect to
        the reference series ('series_from').
        When it is set to be False, the amplitude variations are computed with respect to
        the target series ('series_to'). It is useful for the plotting between a pair of time series,
        when we have more interest in how the target series differs from the reference series.
        :return:
        """
        if on_segments:
            path = self.segments_to_path()
        else:
            path = self.path
        series_from = self.series_from if amplitude_on_series_from else self.series_to
        series_to = self.series_to if amplitude_on_series_from else self.series_from
        path = path if amplitude_on_series_from else [(b, a) for a, b in path]
        if self.ndim == 1:
            variations = np.zeros((len(series_from), 2))
            tvalues = defaultdict(lambda: ([], []))  # todo: no need to keep all values
            for fi, ti in path:
                v = series_to[ti] - series_from[fi]
                if v <= 0:
                    tvalues[fi][0].append(-v)
                if v >= 0:
                    tvalues[fi][1].append(v)
            for fi, (valuesn, valuesp) in tvalues.items():
                if len(valuesn) > 0:
                    varn = max(valuesn)
                else:
                    varn = 0
                if len(valuesp) > 0:
                    varp = max(valuesp)
                else:
                    varp = 0
                variations[fi] = [varn, varp]
        else:
            variations = np.zeros((len(series_from), 2, self.ndim))
            for i_dim in range(self.ndim):
                tvalues = defaultdict(lambda: ([], []))  # todo: no need to keep all values
                for fi, ti in path:
                    v = series_to[ti, i_dim] - series_from[fi, i_dim]
                    if v <= 0:
                        tvalues[fi][0].append(-v)
                    if v >= 0:
                        tvalues[fi][1].append(v)
                for fi, (valuesn, valuesp) in tvalues.items():
                    if len(valuesn) > 0:
                        varn = max(valuesn)
                    else:
                        varn = 0
                    if len(valuesp) > 0:
                        varp = max(valuesp)
                    else:
                        varp = 0
                    variations[fi,:,i_dim] = [varn, varp]
        return variations

    def get_bounds(self, on_segments=False):
        """Compute the amplitude bounds

        :param on_segments: Compute the bounds based on the linear segments
            instead of the original optimal path
        :return:
        """
        if on_segments:
            path = self.segments_to_path()
        else:
            path = self.path
        relbounds = np.zeros((len(self.series_from), 2))
        tvalues = defaultdict(lambda: list())
        for fi, ti in path:
            v = self.series_to[ti] - self.series_from[fi]
            tvalues[fi].append(v)
        for fi, values in tvalues.items():
            if len(values) > 0:
                varn = -min(values)
                varp = max(values)
            else:
                varn = 0
                varp = 0
            relbounds[fi] = [varn, varp]
        return relbounds

    def to_h5(self, filename):
        import h5py
        assert self.segments is not None
        with h5py.File(filename, 'w') as f:
            assert self.variations_on_segments
            if hasattr(self, 'series_from_orig'):
                series_from = self.series_from_orig
            else:
                series_from  = self.series_from
            if hasattr(self, 'series_to_orig'):
                series_to = self.series_to_orig
            else:
                series_to  = self.series_to
            f.create_dataset('series_from', data=np.array(series_from))
            f.create_dataset('series_to', data=np.array(series_to))
            if self.path is not None:
                f.create_dataset('path', data=np.array(self.path))

            self.approx_settings.to_h5_group(f.create_group('approx_settings'))
            self.dtw_settings.to_h5_group(f.create_group('dtw_settings'))

            f.attrs['nb_segments'] = len(self.segments)
            f.create_dataset('s_idx', data=np.array([s.s_idx for s in self.segments]))
            f.create_dataset('e_idx', data=np.array([s.e_idx for s in self.segments]))
            data = np.array([s.s_idx_p if s.s_idx_p is not None else -1 for s in self.segments], dtype=np.int_)
            f.create_dataset('s_idx_p', data=data)
            data = np.array([s.e_idx_p if s.e_idx_p is not None else -1 for s in self.segments], dtype=np.int_)
            f.create_dataset('e_idx_p', data=data)
            f.create_dataset('s_idx_y', data=np.array([s.s_idx_y for s in self.segments]))
            f.create_dataset('e_idx_y', data=np.array([s.e_idx_y for s in self.segments]))
            f.create_dataset('angle', data=np.array([s.angle for s in self.segments]))
            f.create_dataset('shift', data=np.array([s.shift for s in self.segments]))
            f.create_dataset('elasticity', data=np.array([s.elasticity for s in self.segments]))
            f.create_dataset('ignore', data=np.array([s.ignore for s in self.segments]))

            f.create_dataset('line2', data=np.array(self.line2))
            f.create_dataset('split_points', data=np.array(self.split_points))

    @staticmethod
    def from_h5(filename):
        import h5py
        from h5py import Dataset
        pair = None
        with h5py.File(filename, 'r') as f:
            series_from = cast(Dataset, f['series_from'])[:]
            series_to = cast(Dataset, f['series_to'])[:]
            path = cast(Dataset, f['path'])[:] if 'path' in f else None
            approx_settings = ApproxSettings.from_h5_group(f['approx_settings'])
            dtw_settings = None
            if 'dtw_settings' in f:
                dtw_settings = DTWSettings.from_h5_group(f['dtw_settings'])
            pair = ExplainPair(
                series_from, series_to,
                path=path,
                auto_run=False,
                dtw_settings=dtw_settings,
                approx_settings=approx_settings,
            )

            # Do not compute segments but load from file
            nb_segments = cast(int, f.attrs['nb_segments'])
            pair.segments = []
            s_idx = cast(Dataset, f['s_idx'])
            e_idx = cast(Dataset, f['e_idx'])
            s_idx_y = cast(Dataset, f['s_idx_y'])
            e_idx_y = cast(Dataset, f['e_idx_y'])
            s_idx_p = cast(Dataset, f['s_idx_p'])
            e_idx_p = cast(Dataset, f['e_idx_p'])
            angle = cast(Dataset, f['angle'])
            shift = cast(Dataset, f['shift'])
            elasticity = cast(Dataset, f['elasticity'])
            ignore = cast(Dataset, f['ignore'])
            for i in range(nb_segments):
                segment = Segment(
                    s_idx[i], e_idx[i],
                    s_idx_y[i], e_idx_y[i],
                    s_idx_p[i], e_idx_p[i],
                    angle[i], shift[i],
                    elasticity[i], ignore[i],
                )
                pair.segments.append(segment)
            pair.line2 = cast(Dataset, f['line2'])[:]
            pair.split_points = cast(Dataset, f['split_points'])[:]

        return pair

    # TODO: deprecate?
    # def plot(self, filename=None, fig=None):
    #     fig, _ = plot_explain(
    #         self.series_from,
    #         self.segments,
    #         self.variations,
    #         filename=filename,
    #         fig=fig,
    #     )
    #     return fig

    def plot_warping(
        self,
        filename=None,
        fig=None,
        axs=None,
        series_line_options=None,
        warping_line_options=None,
        elasticity_line_options=None,
        show_xticks=True,
        show_yticks=True,
        tick_kwargs=None,
        show_amplitude=True,
        amplitude_on_series_from=False,
        show_elasticity=True,
        color_elasticity: bool=True,
        show_compact_ratio: bool=True,
        nice_layout=True,
        show_legend=True,
    ):
        if show_amplitude:
            variations = self.get_variations(on_segments=True, amplitude_on_series_from=amplitude_on_series_from)
        else:
            variations = None
        assert self.segments is not None
        return plot_warping(
            s1 = self.series_from,
            s2 = self.series_to,
            segments = self.segments,
            path = self.path,
            filename = filename,
            fig = fig,
            axs = axs,
            series_line_options = series_line_options,
            warping_line_options = warping_line_options,
            elasticity_line_options = elasticity_line_options,
            show_xticks = show_xticks,
            show_yticks = show_yticks,
            tick_kwargs = tick_kwargs,
            variations = variations,
            show_elasticity = show_elasticity,
            ndim = self.ndim,
            show_legend=show_legend,
            color_elasticity = color_elasticity,
            show_compact_ratio = show_compact_ratio,
            nice_layout = nice_layout,
            amplitude_on_series_from = amplitude_on_series_from,
        )

    # TODO: deprecate?
    # def plot_explanation_and_warping(self, filename=None, relsize=0.4):
    #     if self.line2[-1][0] != self.line2[-1][1]:
    #         raise AttributeError(f"Not supported for two series of different lengths "
    #                              f"({self.line2[-1][0]} != {self.line2[-1][1]})")
    #     import matplotlib.pyplot as plt
    #
    #     fig = plt.figure()
    #     fig, gs = plot_explain(
    #         self.series_from, self.segments, self.variations, fig=fig
    #     )
    #     assert fig is not None
    #     gs.update(top=0.95, bottom=relsize + 0.05)
    #     new_gs = fig.add_gridspec(2, 1, top=relsize - 0.05, bottom=0.05)
    #     axs = [fig.add_subplot(new_gs[0, 0]), fig.add_subplot(new_gs[1, 0])]
    #     axs[0].set_xlim(-5, len(self.series_from) + 5)
    #     axs[0].set_title("Warped from time series")
    #     axs[1].set_xlim(-5, len(self.series_from) + 5)
    #     axs[1].set_title("Warped to time series")
    #     self.plot_warping(fig=fig, axs=axs)
    #     # fig.tight_layout()
    #
    #     if filename is not None:
    #         fig.savefig(filename)
    #         plt.close(fig)
    #         fig = None
    #     return fig

    def plot_segments(self, filename=None, show_values=False, cost_matrix = None, path_kwargs=None, matshow_kwargs=None, showdist=True,
                      tick_kwargs=None, dsw_path_kwargs=None, figure=None, show_diagonal=False):
        import matplotlib.pyplot as plt
        ya, yb = self.series_from, self.series_to
        path = np.array(self.line2)
        dist, paths = warping_paths(ya, yb, **self.dtw_settings.kwargs())
        dist_approx = self.distance_approx()
        fig, axs = dtwvis.plot_warpingpaths(ya, yb, paths, path=self.path, path_kwargs=path_kwargs,
                                            cost_matrix=cost_matrix,
                                            matshow_kwargs=matshow_kwargs, tick_kwargs=tick_kwargs,
                                            figure=figure, show_diagonal=show_diagonal)
        if dsw_path_kwargs is None:
            dsw_path_kwargs = {
                'linestyle': '-', 'marker': 'o', 'alpha': 0.5,
                'color': 'green', 'linewidth': 3
            }
        axs[0].plot(path[:, 1], path[:, 0], **dsw_path_kwargs)

        if showdist:
            if self.approx_settings.warp_penalty is not None:
                dist_approx_penalty = self.distance_approx(with_warp_penalty=True)
                axs[3].text(0, 0.6, f"DSW p = {dist_approx_penalty:.3f}")
                axs[3].text(0, 0.4, f"DSW = {dist_approx:.3f}")
                dist_penalty = self.distance(with_warp_penalty=True)
                axs[3].text(0, 0.2, f"Dist p = {dist_penalty:.3f}")
            else:
                axs[3].text(0, 0.2, f"Dist DSW = {dist_approx:.3f}")

        if show_values:
            # Print segment statistics
            if self.segments is None:
                raise Exception("No segments available for the show_values=True argument")

            if self.ndim == 1:
                # top time series
                points1 = [(s.s_idx_y, self.series_to[s.s_idx_y]) for s in self.segments]
                points1.append((self.segments[-1].e_idx_y, self.series_to[self.segments[-1].e_idx_y]))
                points1 = np.array(points1)
                axs[1].scatter(points1[:,0], points1[:,1], marker="+", color='k')

                # left time series
                points2 = [(-self.series_from[s.s_idx], s.s_idx) for s in self.segments]
                points2.append((-self.series_from[self.segments[-1].e_idx], self.segments[-1].e_idx))
                points2 = np.array(points2)
                axs[2].scatter(points2[:,0], points2[:,1], marker="+", color='k')

                # Show on other time series to simplify comparison in amplitued
                axs[1].scatter(points1[:,0], -points2[:,0], marker="+", color='k', alpha=0.5)
                axs[2].scatter(-points1[:,1], points2[:,1], marker="+", color='k', alpha=0.5)

            # cost plot
            with_warp_penalty = False if self.warp_penalty_cost is None or self.warp_penalty_cost == 0 else True
            rdist, rdists = self.distance_per_segment()
            adist, adists = self.distance_approx_per_segment()
            satisfies_all, satisfied = self.check_is_local_tolerance_satisfied()
            for segment, srdist, sadist, satisf in zip(self.segments, rdists, adists, satisfied):
                annotation = (f"{srdist:.4f} / {sadist:.4f} - "
                              f"({segment.s_idx},{segment.s_idx_y}) - [{segment.s_idx_p}]")
                color = dsw_path_kwargs.get("color", "green") if satisf else "red"
                axs[0].text((segment.s_idx_y + segment.e_idx_y)/2 + 4,
                            (segment.s_idx + segment.e_idx) / 2,
                            annotation,
                            color=color)
            if with_warp_penalty:
                _, rdistsp = self.distance_per_segment(with_warp_penalty=with_warp_penalty)
                _, adistsp = self.distance_approx_per_segment(with_warp_penalty=with_warp_penalty)
                for segment, srdist, srdistp, sadist, sadistp in zip(self.segments, rdists, rdistsp, adists, adistsp):
                    annotation = (f"{srdist:.4f} / {sadist:.4f} - "
                                  f"{srdistp:.4f} / {sadistp:.4f} - "
                                   f"({segment.s_idx},{segment.s_idx_y}) - [{segment.s_idx_p}]")
            # Show split metric using a circles
            split_strategy = self.approx_settings.split_strategy
            if split_strategy == SplitStrategy.DERIV:
                path_val = np.array(self.path)
                inner_dist, _, _ = self.dtw_settings.inner_dist_fns()
                path_ders = dsw_utils.get_2ndderiv_in_path(self.series_from, self.series_to, self.path)
                path_ders = 200 * (path_ders / np.max(path_ders))
                axs[0].scatter(path_val[:, 1], path_val[:, 0], s=path_ders, c='red', alpha=0.2)
            elif split_strategy == SplitStrategy.DERIV_DIST:
                path_val = np.array(self.path)
                inner_dist, _, _ = self.dtw_settings.inner_dist_fns()
                path_ders1 = dsw_utils.get_1stderiv_in_path(self.series_from, self.series_to, self.path)
                path_ders2 = dsw_utils.get_2ndderiv_in_path(self.series_from, self.series_to, self.path)
                mean = np.mean(path_val, axis=1)
                dists = np.linalg.norm(path_val - mean[:, np.newaxis], axis=1)
                path_ders = dists*path_ders1 + dists**2*path_ders2
                # path_ders = 200 * (path_ders / np.max(path_ders))
                # Inset plot
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_inset = inset_axes(axs[0], width="60%", height="15%", loc='upper right')
                ax_inset.set_facecolor('#808080')
                ax_inset.vlines([s.s_idx_p for s in self.segments], 0, np.max(path_ders1), color='w', alpha=0.1)
                ax_inset.plot(path_ders1, color='orange', alpha=0.5, label='1st')
                ax_inset.plot(path_ders2, color='yellow', alpha=0.5, label='2nd')
                inner_dist, _, _ = self.dtw_settings.inner_dist_fns()
                costs = np.empty(len(self.path))
                for idx, (i, j) in enumerate(self.path):
                    costs[idx] = inner_dist(self.series_from[i], self.series_to[j])
                ax_inset.plot(costs, color='cyan', alpha=0.5, label='cost')
                ax_inset.set_title('1st and 2nd derivative of path', fontsize=8)
                ax_inset.legend(loc='upper right')
                ax_inset.set_xlim((0, len(path_ders1)))

                # Scatter plot
                path_ders12_max = max(np.max(path_ders1), np.max(path_ders2))
                path_ders1 = 400 * (path_ders1 / path_ders12_max)
                path_ders2 = 400 * (path_ders2 / path_ders12_max)
                # axs[0].scatter(path_val[:, 1], path_val[:, 0], s=path_ders, c='red', alpha=0.2)
                axs[0].scatter(path_val[:, 1], path_val[:, 0], s=path_ders2, marker='o',
                               facecolors='none', edgecolors='gold', alpha=0.3)
                axs[0].scatter(path_val[:, 1], path_val[:, 0], s=path_ders1, marker='o',
                               facecolors='none', edgecolors='orange', alpha=0.3)
            # elif self.split_strategy == SplitStrategy.TS_DIFF:
            #     path_val = np.array(self.path)
            #     # inner_dist, _, _ = innerdistance.inner_dist_fns(self.dtw_settings.inner_dist)
            #     # diffs = self.all_ts_diff(path_val, 0, path_val.shape[0]-1)
            #     # diffs = 200 * (diffs / np.max(diffs))
            #     # axs[0].scatter(path_val[:, 1], path_val[:, 0], s=diffs, c='red', alpha=0.2)
            #     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            #     ax_inset = inset_axes(axs[0], width="60%", height="15%", loc='upper right')
            #     ax_inset.set_facecolor('#808080')
            #     # ax_inset.vlines([s.s_idx_p for s in self.segments], 0, np.max(diffs), color='w', alpha=0.1)
            #     # ax_inset.plot(diffs, color='orange', alpha=0.5, label='ts_diffs')
            #     costs = [(self.series_from[i]-self.series_to[j])**2 for i, j in self.path]
            #     ax_inset.vlines([s.s_idx_p for s in self.segments], 0, np.max(costs), color='w', alpha=0.1)
            #     ax_inset.plot(costs, color='orange', alpha=0.5, label='ts_diffs')


        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
            fig, axs = None, None
        return fig, axs

    def plot_one_segment(self, index_of_intermediate, before_split=True, tick_kwargs=None, filename=None, cost_matrix=None, matshow_kwargs=None,  loc='lower left'):
        import matplotlib.pyplot as plt
        assert (
            self.save_intermediates
            and self.intermediates is not None
            and index_of_intermediate < len(self.intermediates)
        )
        start, end, mid = self.intermediates[index_of_intermediate]
        if not before_split:
            assert mid is not None
        dist, paths = warping_paths(self.series_from, self.series_to)
        path = self.path
        fig, axs = dtwvis.plot_warpingpaths(
            self.series_from, self.series_to, paths, path=path,
            path_kwargs={'color': "#FF9999", 'alpha': 0.3}, matshow_kwargs={'alpha': 0},
            tick_kwargs=tick_kwargs)
        py, px = zip(*path)
        axs[0].plot(px[start:end], py[start:end], ".-", color="red", label='Optimal path')

        if before_split:  # check distance
            axs[0].plot([path[start][1], path[end][1]],
                        [path[start][0], path[end][0]], '-o', alpha=1, color='gray', linewidth=5, zorder=1,
                        label='Segment')

        else:  # find point and split
            axs[0].scatter(path[mid][1], path[mid][0], s=200, color='red', zorder=2, label='Splitting point')  # point
            # F = perpendicular_foot([path[mid][1], path[mid][0]], [path[start][1], path[start][0]],
            #                        [path[end][1], path[end][0]])  # perpendicular foot
            # axs[0].plot([path[mid][1], F[0]], [path[mid][0], F[1]], '-.', alpha=1, color = 'gray', lw=2)  # perpendicular line
            # right angle next to the interset
            # axs[0].plot()
            axs[0].plot([path[start][1], path[end][1]],
                        [path[start][0], path[end][0]], '-o', alpha=1, color='gray',
                        linewidth=5, zorder=1, label='Segment')
            axs[0].plot([path[start][1], path[mid][1]],
                        [path[start][0], path[mid][0]], '-o', alpha=1, color='cyan',
                        linewidth=5, zorder=1, label='Subsegment')
            axs[0].plot([path[mid][1], path[end][1]],
                        [path[mid][0], path[end][0]], '-o', alpha=1, color='cyan',
                        linewidth=5, zorder=1)
        axs[0].scatter(path[start][1], path[start][0], s=100, color='black', zorder=2)
        axs[0].scatter(path[end][1], path[end][0], s=100, color='black', zorder=2)
        axs[3].set_visible(False)
        axs[0].legend(fontsize=30, loc=loc)
        if cost_matrix is not None:
            if matshow_kwargs is None:
                matshow_kwargs = {}
            axs[0].matshow(cost_matrix[1:, 1:], aspect='equal', **matshow_kwargs)
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
            fig, axs = None, None
        return fig, axs

    def plot_simplifiedpath(self, filename=None):
        """Plot the optimal warping path and the simplified path used to compute segments."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        line = np.array(self.path)
        assert self.line2 is not None
        line2 = self.line2
        ax.plot(line[:, 0], line[:, 1], '-')
        ax.plot(line2[:, 0], line2[:, 1], '-o', alpha=0.8)
        ax.plot([0, line[-1][0]], [0, line[-1][1]], alpha=0.3)
        if line[-1][0] != line[-1][1]:
            ax.plot([0, line[-1][0]], [0, line[-1][0]], linestyle='dotted', alpha=0.3)
        ax.set_xlabel("From")
        ax.set_ylabel("To")
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
            fig = None
        return fig

    def plot_local_dtw_path(self, p0, p1, fn):
        import dtaidistance.dtw_visualisation as dtwvis
        cost, dtw_local_path, len = self.get_local_dtw_cost(p0, p1)
        fig, axs = dtwvis.plot_warping(self.series_from, self.series_to, dtw_local_path)
        assert axs is not None
        axs[0].text(0.01, 0.99, f"DTW: {cost:.2f}, len: {len -1}", transform=axs[0].transAxes, va="top", ha="left")
        import matplotlib.pyplot as plt
        plt.savefig(fn)

    def plot_local_dsw_path(self, p0, p1, fn):
        inner_dist, _, _ = self.dtw_settings.inner_dist_fns()
        cost, len_p, dsw_local_path = self._line_cost_with_path(p0, p1, inner_dist = inner_dist, include_begin=False, include_end=True)
        dsw_local_path.insert(0, (p0[0], p0[1])) # the first point is shown on the plotting but not counted in cost and length.
        import dtaidistance.dtw_visualisation as dtwvis
        _, axs = dtwvis.plot_warping(self.series_from, self.series_to, dsw_local_path)
        assert axs is not None
        axs[0].text(0.01, 0.99, f"DSW: {cost:.2f}, len: {len_p}", transform=axs[0].transAxes, va="top", ha="left")
        import matplotlib.pyplot as plt
        plt.savefig(fn)

    def get_local_dtw_cost(self, p0, p1):
        p0_index = self.path.index(tuple(p0))
        p1_index = self.path.index(tuple(p1))
        dtw_local_path = self.path[p0_index:p1_index + 1]
        cost = 0
        inner_dist, _, _ = self.dtw_settings.inner_dist_fns()
        for p in dtw_local_path[1:]: #the first point is counted in previous segment
            cost += inner_dist(self.series_from[p[0]], self.series_to[p[1]])
        return cost, dtw_local_path, len(dtw_local_path)

    def plot_local_dsw_of_two_subpaths(self, p0, p1, p2, fn):
        inner_dist, _, _ = self.dtw_settings.inner_dist_fns()
        cost_left, len_p_left, dsw_local_path_left = self._line_cost_with_path(p0, p1, inner_dist = inner_dist, include_begin=False, include_end=True,)
        cost_right, len_p_right, dsw_local_path_right = self._line_cost_with_path(p1, p2,  inner_dist = inner_dist, include_begin=False, include_end=True)

        dsw_local_path_left.insert(0, (p0[0], p0[1]))

        cost_left_dtw, _, len_left_dtw = self.get_local_dtw_cost(p0, p1)
        cost_right_dtw, _, len_right_dtw = self.get_local_dtw_cost(p1, p2)
        import dtaidistance.dtw_visualisation as dtwvis
        fig, axs = dtwvis.plot_warping(self.series_from, self.series_to, dsw_local_path_left+dsw_local_path_right)
        assert fig is not None and axs is not None
        from matplotlib.patches import ConnectionPatch
        con = ConnectionPatch(xyA=(float(p1[0]), float(self.series_from[p1[0]])), coordsA=axs[0].transData,
                              xyB=(float(p1[1]), float(self.series_to[p1[1]])), coordsB=axs[1].transData,  linewidth=2) # middle point
        fig.add_artist(con)
        # python
        axs[0].text(0.01, 0.99, f"DTW: {cost_left_dtw:.2f}, len: {len_left_dtw - 1}", transform=axs[0].transAxes, va="top", ha="left")
        axs[0].text(0.01, 0.94, f"DSW: {cost_left:.2f}, len: {len_p_left}", transform=axs[0].transAxes, va="top", ha="left")
        axs[0].text(0.99, 0.99, f"DTW: {cost_right_dtw:.2f}, len: {len_right_dtw - 1}", transform=axs[0].transAxes, va="top", ha="right")
        axs[0].text(0.99, 0.94, f"DSW: {cost_right:.2f}, len: {len_p_right}", transform=axs[0].transAxes, va="top", ha="right")
        import matplotlib.pyplot as plt
        plt.savefig(fn)

def plot_stats_vs_delta_rel(s1, s2, delta_rel_list,  fn, delta_abs_ratio=0.5, split_strategy=SplitStrategy.TS_DIFF):
    """
    Visualize the DSW's sensitivity to different delta_rel values.
    :param s1: series from
    :param s2: series to
    :param delta_rel_list: different delta_rel values
    :param fn: the filename to save the plot
    :param delta_abs_ratio: the ratio to compute delta_abs based on DTW distance
    :param split_strategy: the split strategy to use
    """

    from dtaidistance.dtw import distance_fast
    dtw_dist = distance_fast(s1, s2)
    delta_abs = delta_abs_ratio * dtw_dist
    segment_cnts = []
    total_variations = []
    dsw_dists = []
    for delta_rel in delta_rel_list:
        ep =  ExplainPair(s1, s2, split_strategy=split_strategy, delta_rel = delta_rel, delta_abs= delta_abs, get_total_variations=True)
        assert ep.segments is not None
        dsw_dists.append(ep.dsw_dist())
        segment_cnts.append(len(ep.segments))
        total_variations.append(ep.total_variations)
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()

    ax1.plot(delta_rel_list, segment_cnts, '-o', label='Segment count', color='tab:blue')
    ax1.set_xlabel('Delta relative')
    ax1.set_ylabel('Segment count', color='tab:blue')
    ax1.set_ylim(bottom=0)
    from matplotlib.ticker import MaxNLocator
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    ax2.plot(delta_rel_list, total_variations, '-o', label='Total variation', color='tab:orange')
    ax2.set_ylabel('Total variation', color='tab:orange')
    ax2.set_ylim(bottom=0)

    ax3 = ax1.twinx()
    # move the spine out and ensure ticks/labels are on the right and visible
    ax3.spines["right"].set_position(("axes", 1.12))
    ax3.plot(delta_rel_list, dsw_dists, '-o', label='DSW dist', color='tab:gray')
    ax3.set_ylabel("DSW dist", color='tab:gray')
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    ax3.tick_params(axis='y', colors='tab:gray')
    ax3.axhline(y=dtw_dist, color='r', linestyle='--', label='DTW dist')
    ax3.set_ylim(bottom=0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='lower left')

    fig.subplots_adjust(right=0.88)
    plt.savefig(fn, bbox_inches='tight')
    plt.close()


def path_to_segments_with_sp_predefined(path, sps):
    """
    Compute segments and variations that explain the warping path between two series, with the splitting points on the series from are predefined/fixed.
    :param path: Warping path
    :param sps: Splitting points on series from
    :return: (segments, the simplified path)

    """
    line2, lidxs = find_simplified_path_with_sps_on_series_from_predefined(path, sps)
    segments = dsw_path_to_segments(line2, lidxs)
    return segments, line2

def find_simplified_path_with_sps_on_series_from_predefined(path, sps):
    """
     Find the simplified path with the splitting points on the series from are predefined/fixed.
    :param path: Warping path
    :param sps: Splitting points on series from
    :return: the simplified path (in which the criterion check is not guaranteed)

    """
    line = np.asarray(path)
    line2 = [(0, 0)]
    lidxs = [0]
    for sp in sps:
        # Find the point in path with series_from index equal to sp,
        # and return the first one if multiple exist
        path_idx = cast(int, np.where(line[:,0] == sp)[0][0])
        line2.append(line[path_idx])
        lidxs.append(path_idx)
    line2.append(line[-1])
    lidxs.append(len(line)-1)
    return line2, lidxs

def dsw_path_to_segments(line2, lidxs):
    """
    :param line2:  points that are reserved in the simplified path
    :param lidxs:  indexes of the points in original path
    :return: segments
    """
    segments = []
    for idx in range(len(lidxs) - 1):
        bp: npt.NDArray[np.int_] = line2[idx]
        ep: npt.NDArray[np.int_] = line2[idx + 1]
        dx = ep[0] - bp[0] + 1
        dy = ep[1] - bp[1] + 1
        if dx == 0:
            a = np.pi / 2
        else:
            a = np.arctan(dy / dx)
        # print(f"{bp}-{ep} : {a:.3f}")
        # Shift based on middle point of segment
        shift = (bp[1] + ep[1]) / 2 - (bp[0] + ep[0]) / 2
        # Shift based on point closest to diagonal
        # denom = bp[1] - ep[1] - bp[0] + ep[0]
        # if np.isclose(denom, 0.0):
        #     t = 0  # parallel, all points are equal
        # else:
        #     t = max(0.0, min(1.0, (bp[1] - bp[0]) / denom))
        # shift = bp[1] + t * (ep[1] - bp[1]) - bp[0] - t * (ep[0] - bp[0])
        expansion = dy - dx

        segment = Segment(bp[0], ep[0], bp[1], ep[1], lidxs[idx], lidxs[idx + 1], a, shift, expansion)
        segments.append(segment)

    return segments


def _draw_vertical_lines(axs, lines, s1, s2, ndim:int, r_c:int, c_c:int, s1_min:int, s1_max:int, s2_min:int, s2_max:int,
                         warping_line_options, warping_line_options_ext, continue_lines=True):
    import matplotlib.patches as mpatches
    con = mpatches.ConnectionPatch(
        xyA=(r_c, s1_min),
        coordsA=axs[0].transData,
        xyB=(c_c, s2_max),
        coordsB=axs[1].transData,
        **warping_line_options,
    )
    lines.append(con)
    if ndim == 1:
        con = mpatches.ConnectionPatch(
            xyA=(r_c, s1[r_c]),
            coordsA=axs[0].transData,
            xyB=(r_c, s1_min),
            coordsB=axs[0].transData,
            **warping_line_options,
        )
        lines.append(con)
        con = mpatches.ConnectionPatch(
            xyA=(c_c, s2_max),
            coordsA=axs[1].transData,
            xyB=(c_c, s2[c_c]),
            coordsB=axs[1].transData,
            **warping_line_options,
        )
        lines.append(con)
        if continue_lines:
            con = mpatches.ConnectionPatch(
                xyA=(r_c, s1_max),
                coordsA=axs[0].transData,
                xyB=(r_c, s1[r_c]),
                coordsB=axs[0].transData,
                **warping_line_options_ext,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=(c_c, s2[c_c]),
                coordsA=axs[1].transData,
                xyB=(c_c, s2_min),
                coordsB=axs[1].transData,
                **warping_line_options_ext,
            )
            lines.append(con)
    else:
        if continue_lines:
            con = mpatches.ConnectionPatch(
                xyA=(r_c, s1_max),
                coordsA=axs[0].transData,
                xyB=(r_c, s1_min),
                coordsB=axs[0].transData,
                **warping_line_options_ext,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=(c_c, s2_max),
                coordsA=axs[1].transData,
                xyB=(c_c, s2_min),
                coordsB=axs[1].transData,
                **warping_line_options_ext,
            )
            lines.append(con)


def plot_warping(
    s1,
    s2,
    segments: list[Segment],
    path=None,
    filename=None,
    fig=None,
    axs=None,
    series_line_options=None,
    warping_line_options=None,
    elasticity_line_options=None,
    start_on_curve=False,
    continue_lines=True,
    show_xticks=True,
    show_yticks=True,
    tick_kwargs=None,
    variations=None,
    show_elasticity=True,
    color_elasticity=False,
    show_compact_ratio=True,
    nice_layout=True,
    amplitude_on_series_from=False,
    ndim=1,
    show_legend=True,
):
    """

    :param s1:
    :param s2:
    :param segments:
    :param path:
    :param filename:
    :param fig:
    :param axs: An array of length 2
    :param series_line_options:
    :param warping_line_options:
    :param elasticity_line_options:
    :param start_on_curve:
    :param continue_lines:
    :param show_xticks:
    :param show_yticks:
    :param tick_kwargs:
    :param variations:
    :param show_elasticity:
    :param color_elasticity:
    :param show_compact_ratio=True:
    :param nice_layout:
    :return: fig, axs
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    if ndim > 1:
        start_on_curve = False
        continue_lines = False

    # Check arguments
    if isinstance(segments, ExplainPair):
        if path is not None:
            raise ValueError("Argument path should be None if segments is of type ExplainPair")
        path = segments.path
        assert segments.segments is not None
        segments = segments.segments
        assert segments is not None
    elif path is None:
        raise AttributeError(
            "Argument path cannot be None if the path argument is not an ExplainPair."
        )
    path = np.asarray(path)
    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex="all", sharey="all")
    elif fig is None or axs is None:
        raise TypeError(
            "The fig and axs arguments need to be both None or both instantiated."
        )

    # Set up axes and plot series
    if series_line_options is None:
        series_line_options = {}
    s1_min = np.min(s1)
    s2_max = np.max(s2)
    s_xlim = max(len(s1) - 1, len(s2) - 1)
    if ndim == 1:
        axs[0].plot(s1, **series_line_options)
        axs[1].plot(s2, **series_line_options)
    else:
        for i_dim in range(ndim):
            axs[0].plot(s1[:,i_dim], **series_line_options)
            axs[1].plot(s2[:,i_dim], **series_line_options)
    if nice_layout:
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['left'].set_position(('outward', 10))
        axs[0].set_xlim(0, s_xlim)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['left'].set_position(('outward', 10))
        axs[1].set_xlim(0, s_xlim)
        if not show_xticks:
            axs[0].set_xticks([])
        if not show_yticks:
            axs[0].set_yticks([])
            axs[1].set_yticks([])
        if tick_kwargs is not None:
            for ax in axs:
                ax.tick_params(**tick_kwargs)
    # plt.tight_layout()

    # Plot amplitude differences
    variations_shade_options = dict(
        color=color_shade,
        alpha=0.8,
        linewidth=0.2,
    )
    if variations is not None:
        ax_to_plot_the_amplitude = axs[0] if amplitude_on_series_from else axs[1]
        s_to_plot_variation_around = s1 if amplitude_on_series_from else s2
        if ndim == 1:
            ax_to_plot_the_amplitude.fill_between(
                range(len(s_to_plot_variation_around)),
                s_to_plot_variation_around - variations[:, 0],
                s_to_plot_variation_around + variations[:, 1],
                **variations_shade_options
            )
        else:
            for i_dim in range(ndim):
                ax_to_plot_the_amplitude.fill_between(
                    range(len(s_to_plot_variation_around)),
                    s_to_plot_variation_around[:,i_dim] - variations[:, 0, i_dim],
                    s_to_plot_variation_around[:,i_dim] + variations[:, 1, i_dim],
                    **variations_shade_options
                )


    # Connection lines
    lines = []
    polys = []
    if warping_line_options is None:
        warping_line_options = {"linewidth": 1, "color": color_shade, "alpha": 0.8}
    warping_line_options_ext = dict(**warping_line_options)
    warping_line_options_ext["linestyle"] = (0, (1, 2))
    warping_line_options_ext["zorder"] = 1
    if elasticity_line_options is None:
        elasticity_line_options = {"facecolor": color_shade, "alpha": 0.4}
    s1_min, s1_max = axs[0].get_ylim()
    s2_min, s2_max = axs[1].get_ylim()

    bbox1 = axs[0].get_position()
    bbox2 = axs[1].get_position()
    dist_between_plots = bbox1.y0 - bbox2.y1
    elasticity_line_height = dist_between_plots / 4

    # Set up colors for elasticity
    if color_elasticity:
        norm = colors.Normalize(vmin=0, vmax=np.pi / 2)
        cmap = plt.get_cmap('seismic')
        angle_to_rgba_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        angle_to_rgba = angle_to_rgba_map.to_rgba
        elasticity_shade_options = {
            "facecolor": color_shade,  # To be override by angle_to_rgba
            "alpha": elasticity_line_options["alpha"]
        }
        warping_line_options["color"] = color_shade_dark
        warping_line_options_ext["color"] = color_shade_dark
        axs[0].set_zorder(1)
        bbox = axs[0].get_position()
        # cax = fig.add_axes((0.70, 0.95, 0.2, 0.02), zorder=10)
        if show_legend:
            cax = fig.add_axes((bbox.x1 - 0.5, bbox.y1 + bbox.height/5, 0.5, bbox.height/15), zorder=10)
            cax.set_facecolor('white')
            cbar = fig.colorbar(angle_to_rgba_map,
                                ticks=[np.arctan(0.1), 0.25 * np.pi, np.arctan(10)],
                                cax=cax, orientation='horizontal', alpha=elasticity_shade_options['alpha'])
            cbar.ax.set_xticklabels(['x0.1', 'x1.0', 'x10'])
            cbar.outline.set_visible(False)
            for tick in cbar.ax.get_xticklabels():
                # tick.set_alpha(alpha=elasticity_shade_options['alpha'])
                # tick.set_verticalalignment("top")
                cbar.ax.tick_params(axis='x', labelsize=10, colors='black')
                bbox_pixels = tick.get_window_extent(renderer=fig.canvas.get_renderer())  # type: ignore
                # bbox_fig = bbox_pixels.transformed(fig.transFigure.inverted())
                # fig.text(
                #     bbox_fig.x0, bbox_fig.y0, "x", ha="right", va="bottom",
                #     fontsize=float(tick.get_fontsize()) / 1.5,
                #     alpha=elasticity_shade_options["alpha"],
                #     transform=fig.transFigure,
                # )
    else:
        angle_to_rgba = None
        elasticity_shade_options = None

    # Plot segments
    if segments is None:
        raise Exception("No segments available")
    for segment in segments:
        assert isinstance(segment, Segment)
        r_c = segment.s_idx
        c_c = segment.s_idx_y
        assert type(path) is list or isinstance(path, np.ndarray)
        assert segment.ignore or (c_c == path[segment.s_idx_p][1])
        assert r_c is not None
        assert c_c is not None
        if r_c < 0 or c_c < 0:
            continue
        if segment.ignore:
            axs[0].axvspan(
                segment.s_idx, segment.e_idx,
                facecolor='none', alpha=0.2, hatch='///', edgecolor='red'
            )
            axs[1].axvspan(
                segment.s_idx_y, segment.e_idx_y,
                facecolor='none', alpha=0.2, hatch='///', edgecolor='red'
            )
        if start_on_curve:
            con = mpatches.ConnectionPatch(
                xyA=(r_c, s1[r_c]),
                coordsA=axs[0].transData,
                xyB=(c_c, s2[c_c]),
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
        else:
            # Use vertical lines to the edge of the plot and only then show the warping.
            _draw_vertical_lines(
                axs, lines, s1, s2, ndim, r_c, c_c, s1_min, s1_max, s2_min, s2_max,
                warping_line_options=warping_line_options,
                warping_line_options_ext=warping_line_options_ext,
                continue_lines=continue_lines,
            )
            if show_elasticity and not color_elasticity:
                if segment.expansion > 0:
                    length = segment.length() - 1
                    coords = [
                        axs[1].transData.transform([c_c + length, s2_max]),
                        axs[1].transData.transform([path[segment.e_idx_p][1], s2_max]),
                        axs[1].transData.transform([path[segment.e_idx_p][1], s2_max]),
                        axs[1].transData.transform([c_c + length, s2_max]),
                    ]
                    coords = fig.transFigure.inverted().transform(coords)
                    coords[2, 1] -= elasticity_line_height
                    coords[3, 1] -= elasticity_line_height
                    poly = mpatches.Polygon(coords, closed=True, transform=fig.transFigure,
                                            **elasticity_line_options)
                    polys.append(poly)
                if segment.compression > 0:
                    length = segment.length() - 1
                    length_y = segment.length_y() - 1
                    coords = [
                        axs[0].transData.transform([segment.e_idx, s1_min]),
                        axs[0].transData.transform([segment.e_idx + length_y - length, s1_min]),
                        axs[0].transData.transform([segment.e_idx + length_y - length, s1_min]),
                        axs[0].transData.transform([segment.e_idx, s1_min]),
                    ]
                    coords = fig.transFigure.inverted().transform(coords)
                    coords[2, 1] += elasticity_line_height
                    coords[3, 1] += elasticity_line_height
                    poly = mpatches.Polygon(coords, closed=True, transform=fig.transFigure,
                                            **elasticity_line_options)
                    polys.append(poly)
            elif show_elasticity and color_elasticity:
                coords = [
                    axs[0].transData.transform([segment.s_idx, s1_min]),
                    axs[0].transData.transform([segment.e_idx, s1_min]),
                    axs[1].transData.transform([segment.e_idx_y, s2_max]),
                    axs[1].transData.transform([segment.s_idx_y, s2_max]),
                ]
                coords = fig.transFigure.inverted().transform(coords)
                assert elasticity_shade_options is not None
                assert angle_to_rgba is not None
                elasticity_shade_options['facecolor'] = angle_to_rgba(segment.angle)  # type: ignore
                poly = mpatches.Polygon(coords, closed=True, transform=fig.transFigure,
                                        **elasticity_shade_options)
                polys.append(poly)
                if show_compact_ratio:
                    compr_ratio = segment.compression_ratio()  # == tan(segment.angle)
                    annotation = f"{compr_ratio:.2f}"
                    x1 = (coords[0][0] + coords[3][0]) / 2
                    x2 = (coords[2][0] + coords[1][0]) / 2
                    text_obj = fig.text((x1 + x2) / 2,
                                        (coords[1][1] + coords[3][1]) / 2 - elasticity_line_height,
                                        annotation, ha='center', va='baseline',
                                        alpha=elasticity_shade_options['alpha'])
                    bbox_text = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())  # type: ignore
                    bbox_poly = poly.get_window_extent(renderer=fig.canvas.get_renderer())  # type: ignore

                    if bbox_text.x0 < bbox_poly.x0 or bbox_text.x1 > bbox_poly.x1:
                        text_obj.remove()
                    else:
                        bbox_fig = fig.transFigure.inverted().transform([(bbox_text.x0, bbox_text.y0)])
                        fig.text(bbox_fig[0][0], (coords[0][1] + coords[3][1]) / 2 - elasticity_line_height,
                                 "x", ha='right', va='baseline',
                                 fontsize=float(text_obj.get_fontsize()) / 1.5,
                                 alpha=elasticity_shade_options['alpha'])

        r_c = segment.e_idx
        c_c = segment.e_idx_y
        assert segment.ignore or (c_c == path[segment.e_idx_p][1])
        if r_c < 0 or c_c < 0:
            continue
        # Also draw last line (at end of segment instead of begin)
        if start_on_curve:
            con = mpatches.ConnectionPatch(
                xyA=(r_c, s1[r_c]),
                coordsA=axs[0].transData,
                xyB=(c_c, s2[c_c]),
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
        else:
            _draw_vertical_lines(
                axs, lines, s1, s2, ndim, r_c, c_c, s1_min, s1_max, s2_min, s2_max,
                warping_line_options=warping_line_options,
                warping_line_options_ext=warping_line_options_ext,
                continue_lines=continue_lines,
            )
    for poly in polys:
        fig.patches.append(poly)
    for line in lines:
        a = fig.add_artist(line)
        a.set_zorder(2)
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, axs = None, None
    return fig, axs


def perpendicular_foot(A, B, C):
    """Find the foot of the perpendicular from A to the line BC."""
    bx, by = B
    cx, cy = C
    ax, ay = A

    # Direction vector of line BC
    BC = np.array([cx - bx, cy - by])

    # Vector from B to A
    BA = np.array([ax - bx, ay - by])

    # Projection formula to find perpendicular foot
    t = np.dot(BA, BC) / np.dot(BC, BC)

    # Compute foot of perpendicular
    foot = np.array(B) + t * BC
    return foot


def rdp_vectorized(points, epsilon, max_num_of_splits=None, scaled_to_same_range = False):
    """
    Ramer-Douglas-Peucker algorithm to simplify a path.

    :param points: 2D numpy array (points x dimensions)
    :param epsilon: Maximum deviation between the original curve and the simplified curve
    """
    original_points = points.copy()
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    if scaled_to_same_range:
        # Scale points to [0, 1] range in both dimensions
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # Prevent division by zero for constant dimensions
        points = (points - min_vals) / ranges

    queue = deque([(0, len(points) - 1)])
    result = set()
    result.add(0)
    result.add(len(points) - 1)

    while queue:
        i0, i1 = queue.popleft()
        p0, p1 = points[i0], points[i1]
        selected_points = points[i0:i1]
        if np.allclose(p0, p1):
            distances = np.linalg.norm(selected_points - p0, axis=1)
        else:
            a = np.abs(
                (selected_points[:, 0] - p0[0]) * (p1[1] - p0[1])
                - (selected_points[:, 1] - p0[1]) * (p1[0] - p0[0])
            )
            p0p1norm = np.linalg.norm(p1 - p0)
            distances = a / p0p1norm

        idxmax = cast(int, np.argmax(distances))
        distmax = distances[idxmax]
        idxmax += i0  # Adjust index to the original index space
        if distmax > epsilon:
            # Keep point
            queue.append((i0, idxmax))
            queue.append((idxmax, i1))
        else:
            # Drop point
            result.add(i0)
            result.add(i1)
            # the first and last points should not be counted as a split but they are in the result set
            if max_num_of_splits is not None and len(result) - 2 >= max_num_of_splits:
                break
    result = sorted(result)
    return remove_singularities(original_points, result)

def remove_singularities(points, result):
    new_points = points[result]

    # Slightly tilt all the vertical segments (to the right if possible, otherwise to the left) by 1 time index.
    if (
            new_points[-2, 0] == new_points[-1, 0]
    ):  # check whether the last segment hints a singularity
        if new_points[-3, 0] == new_points[-2, 0] - 1:
            result.pop(-2)
        else:
            result[-2] = result[-2] - 1  # replacement

    for i in np.arange(
            len(result) - 2, 0, -1
    ):  # check the other segments. It is done reversely since there might be removals on the list.
        if new_points[i, 0] == new_points[i - 1, 0]:
            if new_points[i + 1, 0] == new_points[i, 0] + 1:
                result.pop(i)
            else:
                result[i] = result[i] + 1

    new_points = points[result]
    return new_points, result


def plot_explain(series_from, segments, variations=None, filename=None, fig=None):
    import matplotlib.pyplot as plt

    process = list(enumerate(segments))
    cur_row = 0
    # shift, expansion, compression
    plot_row = np.zeros((len(segments), 3))
    while len(process) != 0:
        s_i = 0
        postpone = []
        for segi, segment in process:
            i0, i1 = segment.s_idx, segment.e_idx
            bi = max(0, min(i0 - segment.shift_l, i0 - segment.expansion))
            ei = max(i1 + segment.shift_r, i1 + segment.expansion)
            if bi >= s_i:
                plot_row[segi, 0] = cur_row
                s_i = ei + 1
            else:
                postpone.append((segi, segment))
        process = postpone
        cur_row += 1

    if fig is None:
        fig = plt.figure()
    gs = fig.add_gridspec(cur_row + 2, 1)

    ax = plt.subplot(gs[0, 0])
    ax.set_title("Series (prototype)")
    ax.set_xticks([])
    ax.set_xlim(-5, len(series_from) + 5)
    ax.plot(series_from, color=color_series)
    ax.vlines(
        [segment.s_idx for segment in segments] + [segments[-1].e_idx],
        ymin=np.min(series_from),
        ymax=np.max(series_from),
        linestyles="dotted",
        color=color_shade,
        alpha=0.2,
    )

    ax = plt.subplot(gs[1: cur_row + 1, 0])
    ax.set_title("Shift + Compression")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-5, len(series_from) + 5)
    ax.vlines(
        [segment.s_idx for segment in segments] + [segments[-1].e_idx],
        ymin=-0.5,
        ymax=cur_row - 0.5,
        linestyles="dotted",
        color=color_shade,
        alpha=0.2,
    )
    seriesp = series_from - np.mean(series_from)
    seriesp = seriesp / (4 * max(np.max(series_from), -np.min(series_from))) + 0.07
    for r in range(cur_row):
        ax.plot(seriesp + r, color=color_series, alpha=0.3)
    max_shift_or_expansion = sys.float_info.min
    min_shift_or_expansion = sys.float_info.max
    for idx in range(len(segments)):
        segment = segments[idx]
        current_max = max(segment.shift_l, segment.shift_r, segment.length() * np.tan(segment.a_compression) / 2,
                          segment.length() * np.tan(segment.a_expansion) / 2)
        current_min = min(segment.shift_l, segment.shift_r, segment.length() * np.tan(segment.a_compression) / 2,
                          segment.length() * np.tan(segment.a_expansion) / 2)
        max_shift_or_expansion = max(max_shift_or_expansion, current_max)
        min_shift_or_expansion = min(min_shift_or_expansion, current_min)

    for idx in range(len(segments)):
        segment = segments[idx]
        bi, ei = segment.s_idx, segment.e_idx
        r = cur_row - plot_row[idx, 0] - 1
        # if r % 2 == 0:
        #     ax.axhspan(r - 0.40, r + 0.60, facecolor=color_bg, alpha=1)

        # Time series segment
        x = [bi, bi, ei, ei, bi]
        y = [r, r + 0.25, r + 0.25, r, r]
        ax.plot(x, y, color=color_shade, alpha=0.9)
        x = [bi, ei]
        y1 = [r + 0.25, r + 0.25]
        y2 = [r, r]
        ax.fill_between(x, y1, y2, color=color_shade, alpha=0.3, linewidth=0)

        # Shift
        x = [bi - segment.shift_l, ei - segment.shift_l]
        y = [r - 0.15, r - 0.15]
        dsw_utils.plot_arrow(x, y, "<-<", segment.shift_l, min_shift_or_expansion, max_shift_or_expansion, ax, color_shade)
        x = [bi + segment.shift_r, ei + segment.shift_r]
        y = [r - 0.3, r - 0.3]
        dsw_utils.plot_arrow(x, y, ">->", segment.shift_r, min_shift_or_expansion, max_shift_or_expansion, ax, color_shade)

        # Compression
        delta = segment.length() * np.tan(segment.a_compression) / 2
        x = [bi + delta, ei - delta]
        y = [r + 0.4, r + 0.4]
        dsw_utils.plot_arrow(x, y, ">-<", delta, min_shift_or_expansion, max_shift_or_expansion, ax, color_shade)

        # Expansion
        delta = segment.length() * np.tan(segment.a_expansion) / 2
        x = [bi - delta, ei + delta]
        y = [r + 0.55, r + 0.55]
        dsw_utils.plot_arrow(x, y, "<->", delta, min_shift_or_expansion, max_shift_or_expansion, ax, color_shade)

    if variations is not None:
        ax = plt.subplot(gs[cur_row + 1, 0])
        ax.set_xlim(-5, len(series_from) + 5)
        ax.vlines(
            [segment.s_idx for segment in segments] + [segments[-1].e_idx],
            ymin=np.min(series_from),
            ymax=np.max(series_from),
            linestyles="dotted",
            color=color_shade,
            alpha=0.2,
        )
        ax.set_title("Amplitude variation")
        ax.plot(series_from, color=color_series)
        ax.fill_between(
            range(len(variations)),
            series_from + variations[:, 1],
            series_from - variations[:, 0],
            color=color_shade,
            alpha=0.3,
            linewidth=0,
        )

    fig.tight_layout()
    if filename is not None:
        fig.savefig(str(filename))
        plt.close(fig)
        fig = None
    return fig, gs


# TODO: deprecate?
# def plot_explain2(series_from, segments, variations=None, filename=None,
#                   fig=None, rel_thr=1,
#                   compact_segments=True, figsize=None,
#                   extra_nrows=0):
#     """Plot the explanation of a prototype wrt all time series used to learn the variation."""
#     import matplotlib.pyplot as plt
#
#     axs = []
#     process = list(enumerate(segments))
#     cur_row = 0
#     # shift, expansion, compression
#     plot_row = np.zeros((len(segments), 3))
#     while len(process) != 0:
#         s_i, e_i, c_i = 0, 0, 0
#         postpone = []
#         for segi, segment in process:
#             i0, i1 = segment.s_idx, segment.e_idx
#             im = int((i0 + i1) / 2)
#             delta_e = segment.length() * (np.tan(diag_angle + rel_thr*segment.a_expansion) - 1) / 2
#             bi = max(0, min(im - segment.shift_l, i0 - delta_e))
#             ei = max(im + segment.shift_r, i1 + delta_e)
#             if bi >= s_i:
#                 plot_row[segi, 0] = cur_row
#                 s_i = ei + 1
#             else:
#                 postpone.append((segi, segment))
#         process = postpone
#         cur_row += 1
#
#     if compact_segments:
#         ratio_gs = math.ceil(len(segments)/4)
#     else:
#         ratio_gs = min(5, cur_row-1)
#     extra_gs = 2 + extra_nrows
#     if fig is None:
#         if figsize is None:
#             figsize = (6, (ratio_gs + 1.5*extra_gs))
#         fig = plt.figure(figsize=figsize)
#     if rel_thr != 1:
#         fig.suptitle(f'Settings: rel_thr={rel_thr}',
#                      x=0.03, y=0.03, ha='left', fontsize=9)
#     gs = fig.add_gridspec(ratio_gs + extra_gs, 1)
#
#     # Prototype
#     ax = plt.subplot(gs[0, 0])
#     axs.append(ax)
#     ax.set_title("Series (prototype)")
#     ax.set_xticks([])
#     ax.set_xlim(-5, len(series_from) + 5)
#     ax.plot(series_from, color=color_series)
#     ax.vlines(
#         [segment.s_idx for segment in segments] + [segments[-1].e_idx],
#         ymin=np.min(series_from),
#         ymax=np.max(series_from),
#         linestyles="dotted",
#         color=color_shade,
#         alpha=0.4,
#     )
#
#     # Segments
#     ax = plt.subplot(gs[1: ratio_gs+1, 0])
#     axs.append(ax)
#     ax.set_title("Shift + Compression")
#     ax.set_yticks([])
#     ax.set_xticks([])
#     ax.set_xlim(-5, len(series_from) + 5)
#     if compact_segments:
#         h = 0.35
#         total_height = len(segments)
#         ax.set_ylim(-0.5, total_height - 0.5)
#     else:
#         h = 0.35
#         total_height = cur_row
#     ax.vlines(
#         [segment.s_idx for segment in segments] + [segments[-1].e_idx],
#         ymin=-0.5,
#         ymax=total_height - 0.5,
#         linestyles="dotted",
#         color=color_shade,
#         alpha=0.4,
#     )
#     if not compact_segments:
#         seriesp = series_from - np.mean(series_from)
#         seriesp = seriesp / (3 * max(np.max(series_from), -np.min(series_from))) + 0.07
#         for r in range(cur_row):
#             ax.plot(seriesp + r, color=color_series, alpha=0.2)
#     for idx in range(len(segments)):
#         segment = segments[idx]
#         bi, ei = segment.s_idx, segment.e_idx
#         if compact_segments:
#             r = total_height - idx - 1
#         else:
#             r = total_height - plot_row[idx, 0] - 1
#         # if r % 2 == 0:
#         #     ax.axhspan(r - 0.40, r + 0.60, facecolor=color_bg, alpha=1)
#
#         # Shift
#         m = (bi + ei) / 2
#         ax.plot([m, m], [r + h, r - h - h / 5], color=color_shade, alpha=0.8, linestyle='solid')
#         x = [m - rel_thr*segment.shift_l, m + rel_thr*segment.shift_r]
#         y = [r - h - h / 5, r - h - h / 5]
#         ax.plot(x, y, color=color_shade, alpha=0.6, linestyle='solid')
#
#         # Time series segment as trapezoid with compression
#         delta_c = (segment.length() * (1 - np.tan(diag_angle - rel_thr*segment.a_compression)) / 2)
#         # print(f'{delta_c*2=} / {segment.compression:.2f} / {segment.length()}')
#         x = [bi, bi + delta_c, ei - delta_c, ei]
#         y1 = [r - h, r + h, r + h, r- h]
#         y2 = [r - h, r -h, r-h, r-h]
#         ax.fill_between(x, y1, y2, color=color_shade, alpha=0.1, linewidth=0)
#         x = [bi, bi + delta_c, ei - delta_c, ei, bi]
#         y = [r-h, r + h, r + h, r-h, r-h]
#         ax.plot(x, y, color=color_shade, alpha=0.6)
#
#         delta_e = (segment.length() * (np.tan(diag_angle + rel_thr*segment.a_expansion) - 1) / 2)
#         # print(f'{delta_e*2=} / {segment.expansion:.2f} / {segment.length()}')
#         x = [bi, bi - delta_e, ei + delta_e, ei]
#         y1 = [r-h, r + h, r + h, r-h]
#         y2 = [r-h, r-h, r-h, r-h]
#         ax.fill_between(x, y1, y2, color=color_shade, alpha=0.1, linewidth=0)
#         x = [bi, bi - delta_e, ei + delta_e, ei, bi]
#         y = [r-h, r + h, r + h, r-h, r-h]
#         ax.plot(x, y, color=color_shade, alpha=0.6)
#
#         if not compact_segments:
#             ax.plot(range(bi, ei+1), seriesp[bi:ei+1] + r, color=color_series, alpha=0.9)
#
#     # Variation
#     if variations is not None:
#         ax = plt.subplot(gs[ratio_gs + 1, 0])
#         axs.append(ax)
#         ax.set_xlim(-5, len(series_from) + 5)
#         ax.vlines(
#             [segment.s_idx for segment in segments] + [segments[-1].e_idx],
#             ymin=np.min(series_from),
#             ymax=np.max(series_from),
#             linestyles="dotted",
#             color=color_shade,
#             alpha=0.4,
#         )
#         ax.set_title("Amplitude variation")
#         ax.plot(range(len(series_from)), series_from, color=color_series)
#         ax.fill_between(
#             range(len(variations)),
#             series_from + rel_thr*variations[:, 1],
#             series_from - rel_thr*variations[:, 0],
#             color=color_shade,
#             alpha=0.5,
#             linewidth=0,
#         )
#
#     # Extra empty rows
#     for i in range(extra_nrows):
#         ax = plt.subplot(gs[ratio_gs + 2 + i, 0])
#         axs.append(ax)
#
#     if filename is not None:
#         fig.tight_layout()
#         fig.savefig(str(filename))
#         plt.close(fig)
#         fig = None
#         gs = None
#         axs = None
#     return fig, gs, axs

