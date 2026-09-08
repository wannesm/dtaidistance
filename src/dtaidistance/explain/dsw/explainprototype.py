# -*- coding: UTF-8 -*-
"""
dtaidistance.explain.dsw.explainprototype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(requires version 2.5.0 higher)

Explain the typical warping path between a set of time series and a prototype
by using Dynamic Subsequence Warping (DSW).

Usage:

::

    dsw_settings = ApproxSettings(
        delta_rel=2,
        delta_abs=0.5,
    )
    dtw_settings = DTWSettings()
    ep = ExplainPrototype(
        y_prototype,
        approx_settings=approx_settings,
        dtw_settings=dtw_settings,
    )
    ep.explain(y_timeseries)
    ep.plot(filename="/path/to/file.png")


:copyright: Copyright 2025-2026 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import math
import logging
from dataclasses import dataclass, asdict, fields
from typing import Optional, Any, Union, cast
from collections import deque

import numpy as np

from .explainpair import ExplainPair, ApproxSettings, Segment, SplitStrategy
from .explainpair import color_series, color_shade, color_series_to, FocusOnShape
from ...dtw import DTWSettings
from ...innerdistance import InnerDistBaseObject
from ...util import SortedList

logger = logging.getLogger("be.kuleuven.dtai.distance")

diag_angle = np.pi / 4


@dataclass
class EPStyle:
    color_prob_s: str = '#b64aae'
    color_prob_e: str = '#4a6aae'
    color_prob_a: str = '#a99448'
    color_shift: str = '0.35'
    color_box: str = 'C0'
    color_compr: str = 'C2'
    color_expan: str = 'C1'
    shift_below: bool = False


def get_epstyle(style:Optional[Union[int, EPStyle]]=1):
    if isinstance(style, EPStyle):
        return style
    if style == 2:
        return EPStyle(
            color_shift = color_shade,
            color_box = color_shade,
            color_compr = color_shade,
            color_expan = color_shade,
            shift_below = True,
        )
    return EPStyle()


@dataclass
class NoiseEstimation:
    """Noise estimation for each parameter.

    The noise should initially be allowed.

    Noise estimation is used to avoid an expected variaton of zero and allow
    for some variation in the learned ranges of normal behavior.
    It adds one extra measurement to the list of measurements with the
    value stored in this dataclass.
    """
    # Use three values that have a range between [0,inf)
    var_p: float = 0  # amplitude variation positive
    var_n: float = 0  # amplitude variation negative
    shift_l: float = 1  # shift left
    shift_r: float = 1  # shift right
    ratio_e: float = 0.1  # ratio expansion
    ratio_c: float = 0.1  # ratio compression

    def to_h5_group(self, group):
        for key, val in asdict(self).items():
            group.attrs[key] = val

    @staticmethod
    def from_h5_group(group):
        kwargs = {}
        for attr in fields(NoiseEstimation):
            attr = attr.name
            if attr in group.attrs:
                kwargs[attr] = group.attrs[attr]
        return NoiseEstimation(**kwargs)

    def set_shift(self, shift):
        self.shift_r = shift
        self.shift_l = shift

    def set_ratio(self, ratio):
        self.ratio_e = ratio
        self.ratio_c = ratio

    def set_var(self, var):
        self.var_p = var
        self.var_n = var

    def correct_for_no_segments(self):
        """If there are no segments found, there is no shift or compression.
        """
        # Keep a small value to avoid division by zero when compuing normality
        self.ratio_e = 0.0001
        self.ratio_c = 0.0001
        self.shift_r = 0.0001
        self.shift_l = 0.0001

    def set_ampl_from_std(self, ts, factor=0.13):
        """Set amplitude noise estimate values using the given series.

        It uses 0.13 * standard deviation to set variation_pos and variation_neg.
        This expresses a range that covers approximately 10% of all points.

        If the time series is z-normalized, this is a noise of +/-0.13.

        :param ts: A time series that is used to settings compute from.
        :param factor:
        """
        ampl = float(factor*np.std(ts))  # range that would cover 10% of all points
        self.var_p = ampl
        self.var_n = ampl

    def set_ampl_from_noise(self, ts, factor=1, window_length=5):
        from scipy.signal import savgol_filter
        smooth = savgol_filter(ts, window_length=window_length, polyorder=3)
        noise = ts - smooth
        ampl = float(factor*np.std(noise))
        self.var_p = ampl
        self.var_n = ampl

    def format(self, fmt="6.3f"):
        return (
            "NoiseEstimation:\n"
            f"Amplitude  (p/n): {self.var_p:{fmt}} / {self.var_n:{fmt}}\n"
            f"Shift      (l/r): {self.shift_l:{fmt}} / {self.shift_r:{fmt}}\n"
            f"Elasticity (e/c): {self.ratio_e:{fmt}} / {self.ratio_c:{fmt}}"
        )

    @staticmethod
    def wrap(ne=None, **kwargs):
        if ne is None:
            ne = NoiseEstimation(**kwargs)
        elif isinstance(ne, NoiseEstimation):
            pass
        elif type(ne) is dict:
            ne = NoiseEstimation(**ne)
        else:
            raise ValueError(f'Unknown argument type for NoiseEstimation: {ne}')
        return ne


@dataclass
class SegmentScores:
    logprobs_s: list
    logprobs_e: list
    logprobs_a: list
    lengths: list
    segments: list
    pair: ExplainPair
    rel_thr: float
    cluster: int = 0


@dataclass
class SegmentAgg:
    # Use agg statistics that have a range in [0,inf)
    s_idx: int  # start index, inclusive
    e_idx: int  # end index, inclusive
    shift_l: float = 0
    shift_r: float = 0
    ratio_e: float = 0  # computed as expsize/origsize - 1
    ratio_c: float = 0  # computed as origsize/comprsize - 1

    def angle_e(self, thr=1):
        return np.arctan(self.ratio_e*thr + 1) - diag_angle

    def angle_c(self, thr=1):
        return np.arctan(self.ratio_c*thr + 1) - diag_angle

    def fmt(self, idx_width=2, shift_width=2):
        return ('SegmentAgg('
                f'idx=({self.s_idx:{idx_width}d},{self.e_idx:{idx_width}d}),'
                f's=(-{self.shift_l:{shift_width+4}.3f},'
                   f'+{self.shift_r:{shift_width+4}.3f}),'
                f'r=(-{self.ratio_c:4.3f},'
                   f'+{self.ratio_e:4.3f}))'
        )

    def length(self):
        return self.e_idx - self.s_idx + 1


class ExplainPrototype:
    def __init__(
        self,
        prototype,
        focus_on_shape: Optional[FocusOnShape] = None,
        approx_settings: Optional[ApproxSettings] = None,
        noise_estimation: Optional[NoiseEstimation] = None,
        dtw_settings=None
    ):
        """Compute segments and its variations that explain the warping paths
        between a prototype and a list of series.

        :param prototype: The time series acting as the prototype
        :param focus_on_shape: Whether to find the path based on the tranformed series or only on the original series.
        :param approx_settings: DSW settings using :class:`ApproxSettings`
        :param noise_estimation: Estimation of the noise in the signal 
            using :class:`NoiseEstimation`. This is used to allow for a minimal
            amount of variation even if the series are all identical for that
            segment of time point. It represents the expected amount of noise
            that is to be tolerated. The more series are given, the less this
            value has impact.
        :param dtw_settings: DTW settings using :class:`DTWSettings`
        """
        self.prototype = prototype
        self._check_ts_format(prototype)
        self.focus_on_shape = FocusOnShape.wrap(focus_on_shape, FocusOnShape.NONE)
        self.approx_settings = ApproxSettings.wrap(approx_settings)
        self.approx_settings.focus_on_shape = FocusOnShape.NONE # We will already apply the transformation on the series in the cluster only once if focus_on_shape is not None.
        self.noise_estimation = NoiseEstimation.wrap(noise_estimation)
        if noise_estimation is None:
            self.noise_estimation.set_ampl_from_std(self.prototype)
        self.dtw_settings = DTWSettings.wrap(dtw_settings)
        self.dtw_settings_tr = self.dtw_settings
        if self.dtw_settings.use_psi():
            # Disable psi-relaxation for prototype
            self.dtw_settings_tr = DTWSettings(**self.dtw_settings.kwargs())
            # self.dtw_settings_tr.psi = 0
            _, _, psi_2b, psi_2e = self.dtw_settings.split_psi()
            # Do not shrink the prototype
            self.dtw_settings_tr.psi = [0, 0, psi_2b, psi_2e]
            self.dtw_settings = self.dtw_settings_tr

        self.estimation_strategy = 'ml'
        self.satisfy_thr = 0.8
        self.pruned_agg = 0.1

        self.variations = None
        self.segments: Optional[list[SegmentAgg]] = None
        # Keep track of the explanation pairs for extended visualization and debugging
        # Not necessary to keep to compute scores and visualizing the prototype with variations.
        self.explanations: Optional[list[ExplainPair]] = None

    def __getstate__(self):
        """Prepare state for pickle."""
        state = self.__dict__.copy()
        # for attr in ['explanations0', 'explanations']:
        for attr in ['explanations0']:
            if attr in state:
                del state[attr]
        return state

    def clear_intermediate_states(self):
        """Remove all attributes that store intermediate datastructures that
        are not required for computing scores.

        This is useful to call before pickling the object to not store
        unecessary information.
        """
        self.explanations = None
        if hasattr(self, 'explanations0'):
            del self.explanations0

    def _check_ts_format(self, ts):
        if not isinstance(ts, np.ndarray):
            raise Exception(f"Time series is not a numpy array: {type(ts)}")
        if len(ts.shape) > 1 and ts.shape[0] == 1:
            raise Exception(f"First dimension is not time: {ts.shape}")

    def explain(self, series, paths=None,
                save_explanations=False):
        """Build the explanation from the prototype based on the given series.

        :param series: List of series
        :param paths: Precomputed path coordinates between prototype and
            instances in the given list of series
        :param save_explanations: For debug purposes, save intermediate steps
        :return: The splitpoints on the prototype.
        """
        if self.focus_on_shape is not None:
            series = self.transform_series(series)
            if self.focus_on_shape in [FocusOnShape.DERIVATIVE]:
                self.prototype_orig = self.prototype
                self.prototype = self.transform_series([self.prototype])[0]

        splitpoints, explanations0 = self.find_all_splitpoints(
            series, paths=paths, save_explanations=save_explanations
        )
        if len(splitpoints) == 0:
            explanations = explanations0
        else:
            explanations = []
            approx_settings_kwargs = self.approx_settings.kwargs()
            del approx_settings_kwargs["split_strategy"]
            for y_idx, pair0 in enumerate(explanations0):
                pair = ExplainPair(
                    self.prototype,
                    pair0.series_to,
                    split_strategy=SplitStrategy.ONLY_INIT_SPLIT_POINTS,
                    init_split_points=splitpoints,
                    **approx_settings_kwargs,
                    dtw_settings=self.dtw_settings_tr,
                    path=pair0.path,
                    variations_on_segments=True,
                    do_remove_singularities=False,
                )
                assert pair.segments is not None
                explanations.append(pair)

        self.split_points = splitpoints
        if save_explanations:
            self.explanations = explanations
            self.explanations0 = explanations0
        else:
            self.explanations = None
            self.explanations0 = None
        self.segments = self.aggregate_segments(explanations)
        self.variations = self.aggregate_points(explanations)

        return splitpoints

    def transform_series(self, series):
        """Apply a transformation on the amplitude of the time series.
        For example, to focus on the shape instead of absolute differences.

        This method uses the focus_on_shape attribute to decide which
        transformation to use.
        """
        if type(series) is list:
            series2 = [None]*len(series)
        elif isinstance(series, np.ndarray):
            series2 = np.empty(series.shape)
        else:
            raise Exception(f"Unknown type for series: {type(series)}")
        if self.focus_on_shape == FocusOnShape.NONE:
            series2 = series
        elif self.focus_on_shape in [FocusOnShape.AFFINEDTW2, FocusOnShape.AFFINEDTW]:
            from ...affinedtw import scale_fast as affinedtw_scale_fast
            # Ignore direction, needs to be consistent with prototype
            for si, s in enumerate(series):
                s2 = affinedtw_scale_fast(s, self.prototype)
                series2[si] = s2
        elif self.focus_on_shape in [FocusOnShape.DERIVATIVE]:
            from ...preprocessing import derivative 
            for si, s in enumerate(series):
                s2 = derivative(s)
                series2[si] = s2  # type: ignore
        else:
            raise ValueError(f"Unknown transformation: {self.focus_on_shape}")
        return series2

    def find_all_splitpoints(self, series, paths=None, save_explanations=False):
        # TODO: check for double operations when calling explainpair multiple times
        # Collect all explanation pairs and used split points
        assert self.prototype is not None
        explanations = []
        for y_idx, y in enumerate(series):
            pair = ExplainPair(
                self.prototype, y,
                **self.approx_settings.kwargs(),
                dtw_settings=self.dtw_settings_tr,
                path=None if paths is None else paths[y_idx],
                variations_on_segments=True,
            )
            assert pair.segments is not None
            assert len(pair.segments) > 0
            explanations.append(pair)
        splitpoints, _ = self.ordered_splitpoints_bycount(explanations)
        if len(splitpoints) == 0:
            return splitpoints, explanations

        # Additional splitpoints to make all segments in all pairs adhere
        # to the local criterium
        explanations2 = []
        all_splitpoints: set[int] = set(splitpoints)
        new_splitpoints = []
        approx_settings_kwargs = self.approx_settings.kwargs()
        del approx_settings_kwargs["approx_prune"]
        for y_idx, pair in enumerate(explanations):
            # TODO: we can overwrite the previous explanations to speed up
            pair2 = ExplainPair(
                self.prototype, pair.series_to,
                init_split_points=splitpoints,
                **approx_settings_kwargs,
                approx_prune=False,
                dtw_settings=self.dtw_settings_tr,
                path=pair.path,
                variations_on_segments=True,
            )
            assert pair2.segments is not None
            # pair2.check_is_local_tolerance_satisfied(verbose=True)
            explanations2.append(pair2)
            for s in pair2.segments[1:]:
                if s.s_idx not in all_splitpoints:
                    all_splitpoints.add(s.s_idx)
                    new_splitpoints.append(int(s.s_idx))
        # least important ones first
        splitpoints = new_splitpoints + list(splitpoints)

        explanations3 = []
        approx_settings_kwargs = self.approx_settings.kwargs()
        del approx_settings_kwargs["split_strategy"]
        for y_idx, pair in enumerate(explanations):
            pair3 = ExplainPair(
                self.prototype, pair.series_to,
                split_strategy=SplitStrategy.ONLY_INIT_SPLIT_POINTS,
                init_split_points=splitpoints,
                **approx_settings_kwargs,
                dtw_settings=self.dtw_settings_tr,
                path=pair.path,
                variations_on_segments=True,
            )
            assert pair3.segments is not None
            # pair3.check_is_local_tolerance_satisfied(verbose=True)
            explanations3.append(pair3)

        queue = deque()
        splitpoints_sorted = SortedList(splitpoints)
        # When a singularity occurs on the first or last position, this is
        # added as a splitpoint, remove it again, in the sorted list there
        # we want to see the start and and end point
        if splitpoints_sorted[0] == 0:
            splitpoints.remove(0)
        else:
            splitpoints_sorted.add(0)
        if splitpoints_sorted[-1] == len(self.prototype)-1:
            splitpoints.remove(len(self.prototype)-1)
        else:
            splitpoints_sorted.add(len(self.prototype)-1)
        sps_fromi = splitpoints_sorted.to_list()
        sps_pathi = [
            # [p.segments[0].s_idx_p] + [s.e_idx_p for s in p.segments]
            p.split_points_pathidx(include_ends=True)
            for p in explanations3
        ]
        assert len(sps_fromi) == len(sps_pathi[0]), f"{sps_fromi} != {sps_pathi[0]}"

        # Try to remove split points one by one ordered by importance
        for i1 in splitpoints:
            queue.appendleft((i1, 0))

        cnt_removals = [0]
        current_lvl = 0
        while len(queue) > 0:
            i1, lvl = queue.pop()
            i0 = splitpoints_sorted.find_lt(i1)
            i2 = splitpoints_sorted.find_gt(i1)
            if lvl > current_lvl:
                if cnt_removals[current_lvl] == 0:
                    break
                cnt_removals.append(0)
                current_lvl = lvl
            try:
                splitpoints_sorted.index(i0)
                splitpoints_sorted.index(i1)
                splitpoints_sorted.index(i2)
            except ValueError:
                continue  # Already removed

            cnt_simplify = 0
            li0, li2 = sps_fromi.index(i0), sps_fromi.index(i2)
            for pairi, pair in enumerate(explanations3):
                pi0, pi2 = sps_pathi[pairi][li0], sps_pathi[pairi][li2]
                pp0, pp2 = pair.path[pi0], pair.path[pi2]
                do_simplify = pair.do_simplify_segment(
                    pi0, pi2,
                    pp0, pp2,
                    pair.ccostv_o,
                    pair.warp_penalty_cost, pair.cpenv_o,
                    pair.tolerance_factor_ab, pair.tolerance_factor_rel,
                    pair.line_cost,
                )
                if do_simplify:
                    cnt_simplify += 1

            # if cnt_simplify == len(explanations3):
            if cnt_simplify/len(explanations3) >= self.satisfy_thr:
                splitpoints_sorted.remove(i1)
                cnt_removals[current_lvl] += 1
            else:
                # try again later
                queue.appendleft((i1, current_lvl + 1))
        return splitpoints_sorted.to_list(), explanations

    def aggregate_points(self, explanations:Optional[list[ExplainPair]] = None):
        if explanations is None:
            explanations = self.explanations
        assert explanations is not None
        # Estimate scale for each pos and neg variation
        all_variations = np.zeros(explanations[0].variations.shape)
        all_counts = np.zeros(explanations[0].variations.shape)
        if self.estimation_strategy == 'max':
            all_variations[:, 0] = self.noise_estimation.var_n
            all_variations[:, 1] = self.noise_estimation.var_p
            for explanation in explanations:
                all_variations = np.maximum(all_variations, explanation.variations)
        else:  # self.estimation_strategy == 'ml'
            all_variations[:, 0] = self.noise_estimation.var_n
            all_variations[:, 1] = self.noise_estimation.var_p
            all_counts[:, :] += 1
            for explanation in explanations:
                variations = explanation.variations
                all_variations += variations
                all_counts[:, :] += np.logical_and(variations[:, 0] == 0, variations[:, 1] == 0)[:, np.newaxis]/2
                all_counts[:, 0] += (variations[:, 0] > 0)
                all_counts[:, 1] += (variations[:, 1] > 0)
            all_variations = np.divide(all_variations, all_counts)
        return all_variations

    def aggregate_segments(self, explanations:Optional[list[ExplainPair]] = None):
        """Aggregate segments and store the exponential scale for each shift and elasticity."""
        if explanations is None:
            explanations = self.explanations
        assert explanations is not None
        assert len(explanations) > 0
        ne = self.noise_estimation
        segments3 = []
        prot_segments = explanations[0].segments_notignored()
        assert prot_segments is not None
        if len(prot_segments) == 1:
            self.noise_estimation.correct_for_no_segments()
        segments_ni = []
        for pair in explanations:
            s = pair.segments_notignored()
            assert s is not None
            segments_ni.append(s)
        for segi in range(len(prot_segments)):
            segment = prot_segments[segi]
            segment3 = SegmentAgg(segment.s_idx, segment.e_idx)
            # Default additional value to avoid a scale of 0
            shift_l, shift_r = [ne.shift_l], [ne.shift_r]
            ratio_e, ratio_c = [ne.ratio_e], [ne.ratio_c]

            shift_l_weight = 1
            shift_r_weight = 1
            ratio_e_weight = 1
            ratio_c_weight = 1
            for pairi, pair in enumerate(explanations):  # todo: no need to maintain six arrays, use six variables should be enough. Cons: codes might be less readable.
                segment = segments_ni[pairi][segi]
                if segment.shift < 0:
                    shift_l.append(segment.shift_l)
                    shift_l_weight += 1
                elif segment.shift > 0:
                    shift_r.append(segment.shift_r)
                    shift_r_weight += 1
                else:
                    shift_l.append(0)
                    shift_r.append(0)
                    shift_l_weight += 0.5
                    shift_r_weight += 0.5

                if segment.elasticity < 0:
                    ratio_c.append(segment.ratio_c)
                    ratio_c_weight += 1
                elif segment.elasticity > 0:
                    ratio_e.append(segment.ratio_e)
                    ratio_e_weight += 1
                else:
                    ratio_c.append(0)
                    ratio_e.append(0)
                    ratio_c_weight += 0.5
                    ratio_e_weight += 0.5

            if self.estimation_strategy == 'max':
                segment3.shift_l = np.max(shift_l)
                segment3.shift_r = np.max(shift_r)
                segment3.ratio_e = np.max(ratio_e)
                segment3.ratio_c = np.max(ratio_c)
            else:  # estimation_strategy == 'ml'
                segment3.shift_l = np.sum(shift_l) / shift_l_weight
                segment3.shift_r = np.sum(shift_r) / shift_r_weight
                segment3.ratio_e = np.sum(ratio_e) / ratio_e_weight
                segment3.ratio_c = np.sum(ratio_c) / ratio_c_weight
            segments3.append(segment3)
        return segments3

    def ordered_splitpoints_bycount(
            self,
            explanations=None,
            keep_intermediate=False
    ) -> tuple[list[int], Any]:
        """Returnst splitpoints in order of importance.
        First one is least important. Last one is most important.
        """
        if explanations is None:
            explanations = self.explanations
        assert explanations is not None
        assert self.prototype is not None
        cnts = np.zeros(len(self.prototype), dtype=int)
        for pair in explanations:
            assert pair.segments is not None
            split_points = pair.split_points_unique()
            # split_points = pair.split_points
            cnts[split_points] += 1
        w = 5
        assert w % 2 == 1
        wh = int((w-1)/2)
        idx = np.linspace(0, len(cnts)-1, len(cnts))
        scnts = np.convolve(cnts, np.ones(w,dtype=int), 'same')
        scnts += cnts
        kernel = np.ones_like(scnts, dtype=np.float32)
        splitpoints = []
        scntspp = scnts
        valid = cnts > 0
        while np.any(valid):
            sp = int(np.argmax(scntspp))
            valid[sp] = False
            splitpoints.append(sp)
            scnts[max(0,sp-wh):min(len(scnts),sp+wh+1)] -= cnts[sp]
            scnts[sp] -= cnts[sp]
            kernel *= 1-1/2**(np.abs(idx-sp))
            scntspp = scnts * kernel * valid
        extra = (cnts, kernel) if keep_intermediate else None
        splitpoints.reverse()
        return splitpoints, extra

    def ordered_splitpoints_bylength(self, explanations=None):
        if explanations is None:
            explanations = self.explanations
        assert explanations is not None
        all_splits = set()
        for pair in explanations:
            assert pair.segments is not None
            all_splits.update([s.s_idx for s in pair.segments[1:]])
        assert self.prototype is not None
        nb_pts = len(self.prototype)
        all_splits = np.array(sorted(all_splits))
        if_merged_end_indices = np.array(all_splits[1:] + [nb_pts-1])
        if_merged_start_indices = np.array([0] + all_splits[:-1])
        if_merged_new_sizes = if_merged_end_indices - if_merged_start_indices
        return [int(split) for split, size in sorted(zip(all_splits, if_merged_new_sizes), key=lambda x: x[1])]

    def score(
        self,
        ts,
        rel_thr=2,
        path=None,
        return_vis=False
    ):
        ts = self.transform_series([ts])[0]
        approx_settings = self.approx_settings.kwargs()
        approx_settings["approx_prune"] = False
        approx_settings["focus_on_shape"] = self.focus_on_shape # the transformation on the new time series should be consistent with the one applied on the cluster time series.
        # approx_settings["approx_prune"] = PruneStrategy.PRUNE_NOINIT  # TODO: change to this setting
        split_points = self.split_points  # use the splitting points that were seleceted and saved from the explanation phase
        pair = ExplainPair(
            self.prototype,
            ts,
            **approx_settings,
            path=path,
            dtw_settings=self.dtw_settings,
            init_split_points = split_points,
            variations_on_segments=True,
        )
        assert pair.segments is not None

        # Scores for shift and elasticity per segment
        probs_s, probs_e, lengths, segments, approximated_agg_segments = self.score_segments(
            pair, rel_thr=rel_thr
        )
        score_s = self.avg_log_loss(probs_s, lengths)
        score_e = self.avg_log_loss(probs_e, lengths)

        # Scores for amplitude
        probs_a = self.score_points(
            pair, ts, rel_thr=rel_thr, segments=segments,
        )
        score_a = self.amplitude_score(probs_a)

        # Aggregate score
        score = score_s + score_e + score_a

        segment_scores = SegmentScores(probs_s, probs_e, probs_a,
                                    lengths, segments,
                                    pair, rel_thr)
        score_parts = (float(score_s), float(score_e), float(score_a))

        if return_vis:
            figs_about_shift_and_elasticity = self.plot_segment_scores(ts, segments, approximated_agg_segments, probs_s, probs_e, lengths, rel_thr, vis_all_aligned_segments_on_cluster=False)
            figs_about_amplitude = self.plot_point_scores(ts, segment_scores, show_all_in_one_fig=False)
            return float(score), score_parts, segment_scores, pair, figs_about_shift_and_elasticity, figs_about_amplitude
        else:
            return float(score), score_parts, segment_scores, pair

    def get_each_prob_loss(self, logprob, length, total_length):
        return np.abs(-np.multiply(length, logprob) / total_length)

    def plot_segment_scores_on_one_segment(self, new_ts, new_segment, approximated_agg_segment, current_probs_s, current_probs_e, current_lengths,
                                           total_lengths, rel_thr=2.0, vis_all_aligned_segments_on_cluster=False, figsize=(14, 10)):

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        def _row_scale(y, amp=0.35):
            y = np.asarray(y, dtype=float)
            y = y - np.mean(y)
            m = np.max(np.abs(y))
            if m < 1e-12:
                return np.zeros_like(y)
            return amp * y / m

        def _find_original_agg_seg_idx(p_s, p_e, segments):
            # locate the approximated agg segment in the original agg segment space
            for i, seg in enumerate(segments):
                if p_s >= seg.s_idx and p_e <= seg.e_idx:
                    return i, (p_s - seg.s_idx) / (seg.e_idx - seg.s_idx), (p_e - seg.s_idx) / (seg.e_idx - seg.s_idx)
            raise ValueError(f"Could not find original agg segment for indices {p_s}, {p_e}")

        def _find_approx_aligned_segments_on_series(original_agg_seg_idx, start_ratio, end_ratio, all_path_segments):
            # find the segments on the series in the cluster that are aligned with the approximated agg segment on the prototype
            cluster_segs_at_agg_seg_idx = [segs[original_agg_seg_idx] for segs in all_path_segments]
            res = []
            for cluster_seg in cluster_segs_at_agg_seg_idx:
                s_idx_y = int(cluster_seg.s_idx_y + start_ratio * (cluster_seg.e_idx_y - cluster_seg.s_idx_y))
                e_idx_y = int(cluster_seg.s_idx_y + end_ratio * (cluster_seg.e_idx_y - cluster_seg.s_idx_y))
                res.append((s_idx_y, e_idx_y))
            return res

        # ---------- fetch data ----------
        assert self.prototype is not None
        prototype = self.prototype
        p_s = approximated_agg_segment.s_idx
        p_e = approximated_agg_segment.e_idx

        n_s = new_segment.s_idx_y
        n_e = new_segment.e_idx_y

        current_score_for_shift = self.get_each_prob_loss(current_probs_s, current_lengths, total_lengths)
        current_score_for_elasticity = self.get_each_prob_loss(current_probs_e, current_lengths, total_lengths)

        shift_l = approximated_agg_segment.shift_l
        shift_r = approximated_agg_segment.shift_r

        seg_len = p_e - p_s + 1
        mid = 0.5 * (p_s + p_e)

        angle_c = approximated_agg_segment.angle_c(rel_thr)
        angle_e = approximated_agg_segment.angle_e(rel_thr)

        delta_c = -seg_len * np.tan(angle_c) / 2
        delta_e = seg_len * np.tan(angle_e) / 2

        # ---------- figure ----------
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        if vis_all_aligned_segments_on_cluster:
            gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.2, 2.8, 2.4])
        else:
            gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 2.4])


        # (1a) Prototype full series
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("Prototype segment")
        x0 = np.arange(len(prototype))

        ax0.plot(x0, prototype, color="0.75", lw=3.0)
        ax0.plot(np.arange(p_s, p_e + 1), prototype[p_s:p_e + 1], color="C0", lw=5.0)

        ax0.axvspan(p_s, p_e, color="C0", alpha=0.18, label="prototype segment")

        ax0.axvline(p_s, color="0.4", ls=":")
        ax0.axvline(p_e, color="0.4", ls=":")
        ax0.legend(loc="upper right", ncol=2, fontsize=9)

        # (1b) Aggregation diagram
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.set_title("Shift + Compression / Expansion")
        ax1.set_yticks([])
        ax1.set_ylim(-0.55, 0.55)

        h = 0.28

        # original segment
        base_poly = Polygon(
            [[p_s, -h], [p_s, h], [p_e, h], [p_e, -h]],
            closed=True, facecolor="C0", edgecolor="C0", alpha=0.15
        )
        ax1.add_patch(base_poly)
        ax1.plot([p_s, p_s, p_e, p_e, p_s], [-h, h, h, -h, -h], color="C0", lw=1.5)

        # shift
        ax1.plot([mid, mid], [0.30, -0.10], color="0.35", lw=1.4)
        ax1.plot([mid - rel_thr * shift_l, mid + rel_thr * shift_r], [-0.10, -0.10],
                 color="0.35", lw=2.0)

        # compression trapezoid
        comp_poly = Polygon(
            [[p_s, -h], [p_s + delta_c, h], [p_e - delta_c, h], [p_e, -h]],
            closed=True, facecolor="C2", edgecolor="C2", alpha=0.12
        )
        ax1.add_patch(comp_poly)
        ax1.plot([p_s, p_s + delta_c, p_e - delta_c, p_e, p_s],
                 [-h, h, h, -h, -h], color="C2", lw=1.5)

        # expansion trapezoid
        exp_poly = Polygon(
            [[p_s, -h], [p_s - delta_e, h], [p_e + delta_e, h], [p_e, -h]],
            closed=True, facecolor="C1", edgecolor="C1", alpha=0.12
        )
        ax1.add_patch(exp_poly)
        ax1.plot([p_s, p_s - delta_e, p_e + delta_e, p_e, p_s],
                 [-h, h, h, -h, -h], color="C1", lw=1.5)

        ax1.axvline(p_s, color="0.5", ls=":")
        ax1.axvline(p_e, color="0.5", ls=":")

        # (2) Cluster series with aligned segments
        if vis_all_aligned_segments_on_cluster:
            ax2 = fig.add_subplot(gs[2, 0])
            ax2.set_title("Aligned segments on cluster series")


            assert self.explanations is not None
            series_list = [exp.series_to for exp in self.explanations]
            all_path_segments = [exp.segments for exp in self.explanations]

            original_agg_seg_idx, start_ratio, end_ratio = _find_original_agg_seg_idx(p_s, p_e, self.segments)
            cluster_aligned_segments = _find_approx_aligned_segments_on_series(original_agg_seg_idx, start_ratio, end_ratio, all_path_segments)
            valid_pairs = [(ts, seg[0], seg[1]) for ts, seg in zip(series_list, cluster_aligned_segments)]

            n_rows = len(valid_pairs)
            if n_rows == 0:
                ax2.text(0.5, 0.5, "No valid aligned segments in `segs_on_series`.",
                         ha="center", va="center", transform=ax2.transAxes)
                ax2.set_axis_off()
            else:
                for row, (ts, s, e) in enumerate(valid_pairs):
                    baseline = n_rows - row - 1
                    ys = _row_scale(ts) + baseline
                    ax2.plot(np.arange(len(ts)), ys, color="0.75", lw=1.0)
                    ax2.plot(np.arange(s, e + 1), ys[s:e + 1], color="C0", lw=2.0)
                    ax2.text(-2, baseline, f"{row}", va="center", ha="right",
                             fontsize=8, color="0.35")
                    ax2.text(e + 2, baseline, f"[{s}, {e}]", va="center", ha="left",
                             fontsize=8, color="0.35")

                ax2.set_yticks([])
                full_len = max(len(prototype), *(len(s) for s in series_list), len(new_ts))

                ax2.set_xlim(0, full_len - 1)
                ax2.set_ylim(-0.75, n_rows - 0.25)
                ax2.set_ylabel("cluster series")
        else:
            ax2 = None

        # (3) New time series only, with matched segment highlighted
        current_axs_index = 3 if vis_all_aligned_segments_on_cluster else 2
        ax3 = fig.add_subplot(gs[current_axs_index, 0], sharex=ax0)
        ax3.set_title(f"New time series with matched segment shift_score:{current_score_for_shift:.2f}, elasticity_score:{current_score_for_elasticity:.2f}")

        x_new = np.arange(len(new_ts))
        ax3.plot(x_new, new_ts, color="0.75", lw=4.0, label="new time series")
        ax3.plot(np.arange(n_s, n_e + 1), new_ts[n_s:n_e + 1],
                 color="C3", lw=6.0, label=f"matched segment [{n_s}, {n_e}]")
        ax3.axvspan(n_s, n_e, color="C3", alpha=0.15)

        ax3.legend(loc="best", fontsize=9)
        if not vis_all_aligned_segments_on_cluster:
            full_len = max(len(prototype), len(new_ts))
        else:
            full_len = len(prototype)
        for ax in [ax0, ax1, ax3]:
            ax.set_xlim(0, full_len - 1)
        ymin = min(np.min(self.prototype), np.min(new_ts))
        ymax = max(np.max(self.prototype), np.max(new_ts))
        ax1.set_ylim(ymin, ymax)
        ax3.set_ylim(ymin, ymax)

        if vis_all_aligned_segments_on_cluster:
            return fig, (ax0, ax1, ax2, ax3)
        else:
            return fig, (ax0, ax1, ax3)

    def plot_segment_scores(self, ts, segments, approximated_agg_segments, probs_s, probs_e, lengths, rel_thr, vis_all_aligned_segments_on_cluster=False):
        """
        To visualize the shift/elasticity score per segment.
        :param ts: the new time series to be explained
        :param segments: a list of (aggregation segment on the cluster, segment on the new time series)
        :param approximated_agg_segments: a list of approximated aggregation segments on the cluster (the ones that are truly used in the computation of shift/elasticity scores)
        :param vis_all_aligned_segments_on_cluster:  only switch to True when we want to visualize the all the aligned segments on the series in the cluster.

        :return: a list of figures, each figure is about one segment.
        """
        import matplotlib.pyplot as plt
        segments_list = list(zip(*segments))
        agg_segments_list = segments_list[0]
        new_segments_list = segments_list[1]

        figs = []
        total_lengths = np.sum(lengths)
        for i in np.arange(len(agg_segments_list)):
            fig, axs = self.plot_segment_scores_on_one_segment(
                ts, new_segments_list[i], approximated_agg_segments[i], probs_s[i], probs_e[i], lengths[i],
                total_lengths, rel_thr=rel_thr, vis_all_aligned_segments_on_cluster=vis_all_aligned_segments_on_cluster)
            figs.append(fig)
            plt.close(fig)
        return figs

    def plot_point_scores(self, ts, segment_scores: SegmentScores, show_all_in_one_fig=False):
        """
        To visualize the amplitude scores per segment.
        :param ts: the new time series to be explained
        :param show_all_in_one_fig: if True, all the point scores of all segments will be visualized in one figure. If False, each segment will be visualized in a separate figure.
        :return: a list of figures, each figure is about one segment if show_all_in_one_fig is False, otherwise the list contains only one figure which visualizes all segments together.
        """
        import matplotlib.pyplot as plt

        if show_all_in_one_fig:
            fig = self.plot_point_scores_on_one_segment(ts, None, segment_scores)
            plt.close(fig)
            return [fig]
        else:
            figs = []
            assert segment_scores.pair.segments is not None
            n_segments = len(segment_scores.pair.segments)
            for i in range(n_segments):
                fig = self.plot_point_scores_on_one_segment(ts, i, segment_scores)
                plt.close(fig)
                figs.append(fig)
            return figs

    def plot_point_scores_on_one_segment(self, ts, seg_idx, segment_scores: SegmentScores, fig_size = (10, 6)):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm

        pair = segment_scores.pair
        probs_a_all = segment_scores.logprobs_a
        rel_thr = segment_scores.rel_thr
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=fig_size)

        ax1.plot(self.prototype, color="0.75", lw=2.0)
        ax2.plot(ts, color="0.75", lw=2.0, label="new time series")

        # ax1.plot(self.prototype, linewidth=5.0)
        # ax2.plot(ts, color='orange', linewidth=3.0)

        assert self.variations is not None
        assert self.prototype is not None
        variation_pos = np.array([v[0] for v in self.variations])
        variation_neg = np.array([v[1] for v in self.variations])
        ax1.fill_between(
            np.arange(len(self.prototype)),
            self.prototype - rel_thr * variation_neg,
            self.prototype + rel_thr * variation_pos,
            color="orange",
            alpha=0.5,
        )

        path_to_use = pair.dsw_path()
        path_pairs = list(path_to_use)
        path_probs = list(probs_a_all)

        # --- colors from anomaly probabilities ---
        abs_probs = [abs(p) for p in probs_a_all if p != 0]
        norm = mcolors.Normalize(vmin=0, vmax=max(abs_probs) if abs_probs else 1)
        cmap = cm.Reds  # type: ignore

        if seg_idx is not None:
            assert pair.segments is not None
            pathseg = pair.segments[seg_idx]
            starting_idx = pathseg.segment.s_idx
            ending_idx = pathseg.segment.e_idx
            starting_idx_on_nts = pathseg.s_idx_y
            ending_idx_on_nts = pathseg.e_idx_y

            #  --- highlight the current matched segment pair ---
            ax1.plot(np.arange(starting_idx, ending_idx + 1), self.prototype[starting_idx: ending_idx + 1], color="C0", lw=4.0)
            ax2.plot(np.arange(starting_idx_on_nts, ending_idx_on_nts + 1), ts[starting_idx_on_nts: ending_idx_on_nts + 1],
                     color="C3", lw=4.0, label=f"matched segment [{starting_idx_on_nts}, {ending_idx_on_nts}]")

            ax1.axvspan(starting_idx, ending_idx, color="C0", alpha=0.15)
            ax2.axvspan(starting_idx_on_nts, ending_idx_on_nts, color="C3", alpha=0.15)

            start_pos = path_pairs.index((starting_idx, starting_idx_on_nts))
            end_pos = path_pairs.index((ending_idx, ending_idx_on_nts))

            region_matches = path_pairs[start_pos:end_pos + 1]
            region_probs = path_probs[start_pos:end_pos + 1]
        else:
            region_matches = list(path_to_use)
            region_probs = list(probs_a_all)

        matched_y_on_ax1 = {}
        matched_prob_on_ax1 = {}

        for (i, j), p in zip(region_matches, region_probs):
            matched_y_on_ax1.setdefault(i, []).append(ts[j])
            matched_prob_on_ax1.setdefault(i, []).append(abs(p))

        overlay_x = []
        overlay_y = []
        overlay_c = []

        for i in sorted(matched_y_on_ax1):
            current_y = max(matched_y_on_ax1[i], key=abs) if self.estimation_strategy == 'max' else np.mean(matched_y_on_ax1[i])
            current_c = max(matched_prob_on_ax1[i]) if self.estimation_strategy == 'max' else np.mean(matched_prob_on_ax1[i])

            overlay_x.append(i)
            overlay_y.append(current_y)
            overlay_c.append(current_c)

        overlay_x = np.array(overlay_x)
        overlay_y = np.array(overlay_y)
        overlay_c = np.array(overlay_c)

        from matplotlib.collections import LineCollection

        y_ts1 = self.prototype[overlay_x]

        matchings = np.stack(
            [
                np.column_stack([overlay_x, y_ts1]),
                np.column_stack([overlay_x, overlay_y]),
            ],
            axis=1
        )

        colors = cmap(norm(overlay_c))
        lc = LineCollection(matchings,colors=colors,linewidths=1,alpha=0.5, zorder=4)  # type: ignore
        ax1.add_collection(lc)

        ymin = min(np.min(self.prototype), np.min(ts))
        ymax = max(np.max(self.prototype), np.max(ts))
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        fig.tight_layout()
        return fig

    def amplitude_score(self, probs_a):
        probs_a = np.sort(probs_a)[:int(self.pruned_agg * len(probs_a))]
        return self.avg_log_loss(probs_a)

    def avg_log_loss(self, logprobs, weights=None):
        """Negative log of the weighted geometric mean of the probabilities."""
        if weights is None:
            return np.abs(-np.sum(logprobs) / len(logprobs))
        return np.abs(-np.sum(np.multiply(weights, logprobs)) / np.sum(weights))

    def score_segments(self, ep:ExplainPair, rel_thr=2):
        """Compute the shift and elasticity probabilities per segment, where
        segments are the intersection of all segments in both the prototype
        and the new time series.

        :param ep: ExplainPair object for the prototype vs new time series
        :param return_segments: Create and store explicit segments.
            Useful for visualizations.
        :return: List of probabilities for shift and elasticity,
            and the lengths of the segments.
            If return_segments is True, also return the segments.
        """
        probs_s, probs_e, lengths = [], [], []
        i_sp, i_sp_c, i_st, i_st_c = 0, 0, 0, 0
        segments = []
        truly_used_segments = []
        if self.segments is None:
            raise Exception("No segments found")
        if ep.segments is None:
            raise Exception("No segments found")

        while i_sp < len(self.segments) or i_st < len(ep.segments):
            i_sp_c = min(i_sp, len(self.segments) - 1)
            i_st_c = min(i_st, len(ep.segments) - 1)
            sp: SegmentAgg = self.segments[i_sp_c]
            st: Segment = ep.segments[i_st_c]

            if sp.s_idx > st.s_idx:
                s_idx = sp.s_idx
                s_idx_y = int(st.s_idx_y + (st.e_idx_y - st.s_idx_y) *
                              (s_idx - st.s_idx) / (st.e_idx - st.s_idx))
            else:
                s_idx = st.s_idx
                s_idx_y = st.s_idx_y

            if sp.e_idx < st.e_idx:
                e_idx = sp.e_idx
                e_idx_y = int(st.s_idx_y + (st.e_idx_y - st.s_idx_y) *
                              (e_idx - st.s_idx) / (st.e_idx - st.s_idx))
            else:
                e_idx = st.e_idx
                e_idx_y = st.e_idx_y

            sp_length = e_idx - s_idx + 1
            st_length = e_idx_y - s_idx_y + 1
            lengths.append(sp_length + st_length)

            shift_l, shift_r = sp.shift_l, sp.shift_r
            # shift moved to this segment
            sp_m = (sp.s_idx + sp.e_idx) / 2
            st_m = ((s_idx + e_idx) / 2,
                    (s_idx_y + e_idx_y) / 2)
            if st_m[0] <= st_m[1] and st_m[0] <= sp_m:
                # right upper
                sp_m_shift = (sp_m, sp_m + rel_thr*sp.shift_r)
                y_m_shift_moved = (st_m[0] - sp_m_shift[0])*np.tan(diag_angle - sp.angle_c(rel_thr)) + sp_m_shift[1]
                shift_r = (y_m_shift_moved - st_m[0]) / rel_thr
            elif st_m[1] >= st_m[0] > sp_m:
                # right lower
                sp_m_shift = (sp_m, sp_m + rel_thr*sp.shift_r)
                y_m_shift_moved = (st_m[0] - sp_m_shift[0])*np.tan(diag_angle + sp.angle_e(rel_thr)) + sp_m_shift[1]
                shift_r = (y_m_shift_moved - st_m[0]) / rel_thr
            elif st_m[1] < st_m[0] <= sp_m:
                # left upper
                sp_m_shift = (sp_m, sp_m - rel_thr*sp.shift_l)
                y_m_shift_moved = (st_m[0] - sp_m_shift[0]) * np.tan(diag_angle + sp.angle_e(rel_thr)) + sp_m_shift[1]
                shift_l = (st_m[0] - y_m_shift_moved) / rel_thr
            else:
                # left lower
                sp_m_shift = (sp_m, sp_m - rel_thr*sp.shift_l)
                y_m_shift_moved = (st_m[0] - sp_m_shift[0]) * np.tan(diag_angle - sp.angle_c(rel_thr)) + sp_m_shift[1]
                shift_l = (st_m[0] - y_m_shift_moved) / rel_thr
            sp_shift = shift_l if st.shift < 0 else shift_r
            st_shift = st_m[1] - st_m[0]

            path = ep.dsw_path()
            s_idx_p = 0 if len(segments) == 0 else segments[-1][1].e_idx_p
            if s_idx_p is not None:
                while s_idx_p < len(path) and path[s_idx_p][0] < s_idx:
                    s_idx_p += 1
                e_idx_p = s_idx_p + 1
                while e_idx_p < len(path) and e_idx_p >= 0 and path[e_idx_p][0] < e_idx:
                    e_idx_p += 1
            else:
                e_idx_p = None
            dx = e_idx - s_idx
            dy = e_idx_y - s_idx_y
            a = np.pi / 2 if dx == 0 else np.arctan(dy / dx)
            expansion = dy - dx
            segment = Segment(s_idx, e_idx, s_idx_y, e_idx_y, s_idx_p, e_idx_p, a, st_shift, expansion, st.ignore)
            segments.append((sp, segment))
            truly_used_segments.append(SegmentAgg(s_idx = segment.s_idx, e_idx = segment.e_idx, shift_l=shift_l, shift_r=shift_r, ratio_e=sp.ratio_e, ratio_c=sp.ratio_c))

            if st.ignore:
                prob_s = 0
                prob_e = 0
            else:
                prob_s = self.normality_logprob(abs(st_shift), sp_shift, rel_thr=rel_thr)
                st_ratio = st.ratio
                sp_ratio = sp.ratio_e if st_ratio > 0 else sp.ratio_c
                prob_e = self.normality_logprob(abs(st_ratio), sp_ratio, rel_thr=rel_thr)
            # print(f"{st_shift:.2f=}, {sp_shift:.2f=} -> {pro:.2fb}")
            # print(f"{st_ratio=:.2f}, {sp_ratio=:.2f} -> {prob:.2f}")
            probs_s.append(prob_s)
            probs_e.append(prob_e)

            if sp.e_idx <= st.e_idx:
                i_sp += 1
            if sp.e_idx >= st.e_idx:
                i_st += 1

        # if self.dtw_settings.use_psi():
        #     psi_pb, psi_pe, psi_tb, psi_te = self.dtw_settings.split_psi()
        #     if self._is_psi_relaxed(segments[0], psi_pb, psi_tb):
        #         print("Dropping first segment")
        #         probs_s[0] = 0
        #         probs_e[0] = 0
        #     if self._is_psi_relaxed(segments[-1], psi_pe, psi_te):
        #         print("Dropping last segment")
        #         probs_s[-1] = 0
        #         probs_e[-1] = 0

        return probs_s, probs_e, lengths, segments, truly_used_segments

    def _is_psi_relaxed(self, segment, psi_p, psi_t):
        # Simplistic interpretation of psi-relaxation
        # Compressing the segment of the prototype or
        # compressing the segment of the scored series is ok
        # Unless the  prototype segment is smaller than psi
        # Segment in prototype is short, always keep
        if segment[0].length() <= psi_p:
            return False
        if (
            segment[1].e_idx - segment[1].s_idx + 1 <= psi_p
            and segment[1].e_idx_y - segment[1].s_idx_y + 1 <= 2
        ):
            return True
        if (
            segment[1].e_idx_y - segment[1].s_idx_y + 1 <= psi_t
            and segment[1].e_idx - segment[1].s_idx + 1 <= 2
        ):
            return True
        return False

    def score_points(self, ep, ts, rel_thr=2, segments=None):
        """Compute the time point probabilities for the amplitude.

        :param ep: ExplainPair object for the prototype vs new time series
        :param ts: the new time series
        :return: List of probabilities with same length as the new time series
        """
        if self.variations is None:
            raise Exception("No variations available")
        path = np.array(ep.dsw_path())
        assert self.prototype is not None
        diffs = ts[path[:, 1]] - self.prototype[path[:, 0]]
        scales = np.multiply(
            diffs <= 0, self.variations[path[:, 0]][:, 0]
        ) + np.multiply(diffs > 0, self.variations[path[:, 0]][:, 1])
        diffs = abs(diffs)
        logprobs = self.normality_logprob(diffs, scales, rel_thr)

        # if self.dtw_settings.use_psi() and segments is not None:
        #     if self._is_psi_relaxed(segments[0]):
        #         logprobs[segments[0][1].s_idx:segments[0][1].e_idx+1] = 0
        #     if self._is_psi_relaxed(segments[-1]):
        #         logprobs[segments[-1][1].s_idx:segments[-1][1].e_idx+1] = 0

        return logprobs

    def normality_logprob(self, x, scale, rel_thr):
        # Assuming exponential distribution
        assert (type(scale) not in [float, int]) or (scale != 0)
        lcdf = rel_thr - np.divide(x, scale)
        return np.minimum(0.0, lcdf)

    def to_h5(self, filename):
        import h5py
        assert self.segments is not None
        with h5py.File(filename, 'w') as f:
            f.attrs['nb_segments'] = len(self.segments)
            self.noise_estimation.to_h5_group(f.create_group('noise_estimation'))
            self.approx_settings.to_h5_group(f.create_group("approx_settings"))
            self.dtw_settings.to_h5_group(f.create_group("dtw_settings"))
            f.attrs['satisfy_thr'] = self.satisfy_thr
            f.attrs['pruned_agg'] = self.pruned_agg
            f.attrs['focus_on_shape'] = self.focus_on_shape.to_int()
            prototype = self.prototype_orig if hasattr(self, 'prototype_orig') else self.prototype
            f.create_dataset("prototype", data=prototype)
            f.create_dataset("variations", data=self.variations)
            f.create_dataset("s_idx",   data=np.array([sa.s_idx for sa in self.segments]))
            f.create_dataset("e_idx",   data=np.array([sa.e_idx for sa in self.segments]))
            f.create_dataset("shift_l", data=np.array([sa.shift_l for sa in self.segments]))
            f.create_dataset("shift_r", data=np.array([sa.shift_r for sa in self.segments]))
            f.create_dataset("ratio_e", data=np.array([sa.ratio_e for sa in self.segments]))
            f.create_dataset("ratio_c", data=np.array([sa.ratio_c for sa in self.segments]))

    @staticmethod
    def from_h5(filename):
        import h5py
        from h5py import Dataset
        ep = None
        with h5py.File(filename, 'r') as f:
            nb_segments = cast(int, f.attrs['nb_segments'])
            try:
                focus_on_shape = FocusOnShape.from_int(f.attrs['focus_on_shape'])
            except KeyError:
                focus_on_shape = None
            try:
                noise_estimation = NoiseEstimation.from_h5_group(f['noise_estimation'])
            except KeyError:
                noise_estimation = None
            try:
                approx_settings = ApproxSettings.from_h5_group(f["approx_settings"])
            except KeyError:
                approx_settings = None
            try:
                dtw_settings = DTWSettings.from_h5_group(f["dtw_settings"])
            except KeyError:
                dtw_settings = None
            prototype = cast(Dataset, f['prototype'])[:]
            variations = cast(Dataset, f['variations'])[:]
            segments = []
            s_idx = cast(Dataset, f['s_idx'])
            e_idx = cast(Dataset, f['e_idx'])
            shift_l = cast(Dataset, f['shift_l'])
            shift_r = cast(Dataset, f['shift_r'])
            ratio_e = cast(Dataset, f['ratio_e'])
            ratio_c = cast(Dataset, f['ratio_c'])
            for i in range(nb_segments):
                segments.append(SegmentAgg(
                    s_idx[i],
                    e_idx[i],
                    shift_l[i],
                    shift_r[i],
                    ratio_e[i],
                    ratio_c[i],
                ))

            ep = ExplainPrototype(
                prototype,
                focus_on_shape=focus_on_shape,
                approx_settings=approx_settings,
                noise_estimation=noise_estimation,
                dtw_settings=dtw_settings,
            )

            # Internal variables
            ep.segments = segments
            ep.variations = variations

            # Internal settings (that could have been changed)
            if 'pruned_agg' in f.attrs:
                ep.pruned_agg = f.attrs['pruned_agg']
            if 'satisfy_thr' in f.attrs:
                ep.satisfy_thr = f.attrs['satisfy_thr']

        return ep

    def plot_paths(self, filename=None, figure=None, rel_thr=2):
        """Plot the average path of the prototype and the expected variations.

        If pair explanations are saved, also plot all the ExplainPairs this
        prototype is based on
        """
        assert self.segments is not None
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib.ticker import NullLocator

        varcolor = 'green'
        thrcolor = 'lightgreen'
        segcolor = 'blue'
        paircolor = 'red'

        pairs = self.explanations
        assert self.prototype is not None
        s1 = self.prototype
        max_s1_y = len(s1) - 1
        min_y = np.min(s1)
        max_y = np.max(s1)
        idxs = set()

        if figure is None:
            fig = plt.figure(figsize=(10, 10), frameon=True)
        else:
            fig = figure
        gs = gridspec.GridSpec(1, 2, wspace=1, hspace=1,
                               left=0, right=1.0, bottom=0, top=1.0,
                               height_ratios=[1],
                               width_ratios=[1,6])

        # Segmented path
        ax0 = fig.add_subplot(gs[0, 1])
        ax0.xaxis.set_ticks_position('bottom')
        ax0.yaxis.set_ticks_position('right')
        ax0.yaxis.set_inverted(True)
        ax0.set_xlim(-0.5, max_s1_y + 0.5)

        if pairs is not None:
            for curpair in pairs:
                assert curpair.segments is not None
                for st in curpair.segments:
                    ax0.plot([st.s_idx_y, st.e_idx_y],
                             [st.s_idx, st.e_idx], '-o', color=paircolor, alpha=0.3)

        for sp in self.segments:
            idxs.add(int(sp.s_idx))
            idxs.add(int(sp.e_idx))
            ax0.plot([sp.s_idx, sp.e_idx],
                     [sp.s_idx, sp.e_idx], '-o', color=segcolor, alpha=0.5)
            sp_m = int((sp.s_idx + sp.e_idx)/2)
            ax0.plot([sp_m - sp.shift_l, sp_m + sp.shift_r], [sp_m] * 2, '-+', color=varcolor, alpha=0.5)
            ax0.plot([sp_m - sp.shift_l, sp_m - rel_thr * sp.shift_l], [sp_m] * 2, '-+', color=thrcolor, alpha=0.5)
            ax0.plot([sp_m + sp.shift_r, sp_m + rel_thr * sp.shift_r], [sp_m] * 2, '-+', color=thrcolor, alpha=0.5)
            x1_prev = x2_prev = [sp.s_idx, sp_m, sp.e_idx]
            for thr, clr in [(1,varcolor), (rel_thr, thrcolor)]:
                # Adapt to current threshold
                angle_e, angle_c = sp.angle_e(thr), sp.angle_c(thr)
                shift_l, shift_r = sp.shift_l*thr, sp.shift_r*thr
                val_ru = (sp.s_idx - sp_m) * np.tan(diag_angle - angle_c) + sp_m + shift_r
                val_rl = (sp.e_idx - sp_m) * np.tan(diag_angle + angle_e) + sp_m + shift_r
                val_rm = sp_m + shift_r
                val_lu = (sp.s_idx - sp_m) * np.tan(diag_angle + angle_e) + sp_m - shift_l
                val_ll = (sp.e_idx - sp_m) * np.tan(diag_angle - angle_c) + sp_m - shift_l
                val_lm = sp_m - shift_l
                # ax0.plot([val_lu, val_lm, val_ll], [sp.s_idx, sp_m, sp.e_idx], '-', color=clr, alpha=0.5)
                # ax0.plot([val_ru, val_rm, val_rl], [sp.s_idx, sp_m, sp.e_idx], '-', color=clr, alpha=0.5)
                ax0.fill_betweenx([sp.s_idx, sp_m, sp.e_idx],
                                  [val_lu, val_lm, val_ll], x1_prev, color=clr, alpha=0.3)
                ax0.fill_betweenx([sp.s_idx, sp_m, sp.e_idx],
                                  x2_prev,[val_ru, val_rm, val_rl], color=clr, alpha=0.3)
                x1_prev = [val_lu, val_lm, val_ll]
                x2_prev = [val_ru, val_rm, val_rl]

        ax0.hlines(list(idxs), 0, max_s1_y, linestyles='dotted', colors='black', alpha=0.3)

        # Time series on left axis
        ax1 = fig.add_subplot(gs[0, 0], sharey=ax0)
        ax1.set_xlim(-max_y, -min_y)
        ax1.set_axis_off()
        ax1.xaxis.set_major_locator(NullLocator())
        ax1.plot(-s1, range(0, max_s1_y + 1), "-", color='#1f77b4')
        ax1.set_ylim(-0.5, max_s1_y + 0.5)
        ax1.invert_yaxis()

        gs.tight_layout(fig, pad=1.0, h_pad=1.0, w_pad=1.0)

        if filename:
            filename = str(filename)
            plt.savefig(filename)
            plt.close()
            fig = None
        return fig

    def plot(
        self, filename=None, segment_scores=None, rel_thr=2, compact_segments=True, figsize=None,
        epstyle=None,
    ):
        assert self.prototype is not None
        assert self.segments is not None
        assert len(self.segments) > 0
        import matplotlib.pyplot as plt
        epstyle = get_epstyle(epstyle)
        prot = self.prototype
        segments = self.segments
        variations = self.variations
        if segment_scores is not None:
            rel_thr = segment_scores.rel_thr

        if epstyle.shift_below:
            import matplotlib.transforms as transforms

        axs = []
        # shift, expansion, compression
        if compact_segments:
            plot_row = np.zeros((len(segments), 3))
            plot_row[:, 0] = range(len(segments))
            cur_row = len(segments)+1
        else:
            plot_row = np.zeros((len(segments), 3))
            cur_row = 0
            process: list[tuple[int,SegmentAgg]] = list(enumerate(segments))
            while len(process) != 0:
                s_i, e_i, c_i = 0, 0, 0
                postpone = []
                for segi, segment in process:
                    i0, i1 = segment.s_idx, segment.e_idx
                    im = int((i0 + i1) / 2)
                    # TODO: ignores threshold
                    angle_e, angle_c = segment.angle_e(rel_thr), segment.angle_c(rel_thr)
                    delta_e = (
                        segment.length() * (np.tan(diag_angle + angle_e) - 1) / 2
                    )
                    bi = max(0, min(im - segment.shift_l, i0 - delta_e))
                    ei = max(im + segment.shift_r, i1 + delta_e)
                    if bi >= s_i:
                        plot_row[segi, 0] = cur_row
                        s_i = ei + 1
                    else:
                        postpone.append((segi, segment))
                process = postpone
                cur_row += 1
        gs_height_blocks = min(1, math.ceil((cur_row-1)/4))
        gs_height_scores = 0 if segment_scores is None else 3
        gs_height = gs_height_blocks + 1 + gs_height_scores
        if figsize is None:
            figsize = (6, gs_height * 2)
        fig = plt.figure(figsize=figsize)
        # hspace is fraction of average axes height
        gs = fig.add_gridspec(gs_height, 1, hspace=0.3)

        # Variation
        ax = fig.add_subplot(gs[0, 0])
        axs.append(ax)
        if variations is not None:
            ax.set_xlim(-5, len(prot) + 5)
            ax.vlines(
                [segment.s_idx for segment in segments] + [segments[-1].e_idx],
                ymin=np.min(prot),
                ymax=np.max(prot),
                linestyles="dotted",
                color=color_shade,
                alpha=0.8,
            )
            ax.set_title("Prototype + amplitude variation", fontsize=8)
            ax.plot(range(len(prot)), prot, color=color_series)
            ax.fill_between(
                range(len(variations)),
                prot + rel_thr * variations[:, 1],
                prot - rel_thr * variations[:, 0],
                color=color_shade,
                alpha=0.8,
                linewidth=0,
            )
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=6)

        # Segments
        ax = fig.add_subplot(gs[1:gs_height_blocks+1, 0])
        axs.append(ax)
        ax.set_title("Shift + Compression", fontsize=8)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(-5, len(prot) + 5)
        if compact_segments:
            h = 0.35
            total_height = len(segments)
            if epstyle.shift_below:
                ax.set_ylim(-1.0, total_height - 0.0)
            else:
                ax.set_ylim(-0.5, total_height - 0.5)
        else:
            h = 0.35
            total_height = cur_row
        ax.vlines(
            [segment.s_idx for segment in self.segments] + [self.segments[-1].e_idx],
            ymin=-0.5,
            ymax=total_height - 0.5,
            linestyles="dotted",
            color=color_shade,
            alpha=0.4,
        )
        if not compact_segments:
            seriesp = prot - np.mean(prot)
            seriesp = seriesp / (3 * max(np.max(prot), -np.min(prot))) + 0.07
            for idx in range(len(segments)):
                segment = segments[idx]
                bi, ei = segment.s_idx, segment.e_idx
                ax.plot(seriesp + idx, color=color_series, alpha=0.2)
                ax.plot(range(bi, ei+1), seriesp[bi:ei+1] + idx, color=color_series, alpha=0.9)

        for idx in range(len(segments)):
            segment = segments[idx]
            bi, ei = segment.s_idx, segment.e_idx
            r = total_height - plot_row[idx, 0] - 1

            # Shift
            m = (bi + ei) / 2
            if epstyle.shift_below:
                x = [m - rel_thr*segment.shift_l, m + rel_thr*segment.shift_r]
                y = [r - h, r - h]
                line, = ax.plot(x, y, color=color_shade, linestyle='solid')
                inv = ax.transData.inverted()
                y0 = inv.transform((0, 0))[1]
                y1 = inv.transform((0, 2 * line.get_linewidth() * fig.dpi / 72))[1]
                dy = y1 - y0
                line.set_ydata(line.get_ydata() - dy)
                ax.plot([m, m], [r + h, r - h - dy], color=epstyle.color_shift, alpha=0.8, linestyle='solid')
            else:
                ax.plot([m, m], [r + 0.30, r - 0.10], color="0.35", lw=1.4)
                ax.plot(
                    [m - rel_thr * segment.shift_l, m + rel_thr * segment.shift_r],
                    [r - 0.10, r - 0.10],
                    color=epstyle.color_shift,
                    lw=0.8,
                )
                h = 0.28

            y_bottom = r - h
            y_top = r + h

            ax.fill_between(
                [bi, bi, ei, ei],
                [y_bottom, y_top, y_top, y_bottom],
                [y_bottom, y_bottom, y_bottom, y_bottom],
                color=epstyle.color_box,
                alpha=0.15,
                linewidth=0,
            )
            ax.plot(
                [bi, bi, ei, ei, bi],
                [y_bottom, y_top, y_top, y_bottom, y_bottom],
                color=epstyle.color_box,
                lw=0.8,
            )

            # Compression trapezoid
            angle_e, angle_c = segment.angle_e(rel_thr), segment.angle_c(rel_thr)
            delta_c = (segment.length() * (1 - np.tan(diag_angle - angle_c)) / 2)

            ax.fill_between(
                [bi, bi + delta_c, ei - delta_c, ei],
                [y_bottom, y_top, y_top, y_bottom],
                [y_bottom, y_bottom, y_bottom, y_bottom],
                color=epstyle.color_compr,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                [bi, bi + delta_c, ei - delta_c, ei, bi],
                [y_bottom, y_top, y_top, y_bottom, y_bottom],
                color=epstyle.color_compr,
                lw=0.8,
            )

            # Expansion trapezoid
            delta_e = (segment.length() * (np.tan(diag_angle + angle_e) - 1) / 2)

            ax.fill_between(
                [bi, bi - delta_e, ei + delta_e, ei],
                [y_bottom, y_top, y_top, y_bottom],
                [y_bottom, y_bottom, y_bottom, y_bottom],
                color=epstyle.color_expan,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                [bi, bi - delta_e, ei + delta_e, ei, bi],
                [y_bottom, y_top, y_top, y_bottom, y_bottom],
                color=epstyle.color_expan,
                lw=0.8,
            )

        # Segment scores
        if segment_scores is not None and len(list(segment_scores.segments)) > 0:
            # Scores
            ax = fig.add_subplot(gs[gs_height_blocks+1, 0])
            axs.append(ax)
            _, sc_segments = zip(*segment_scores.segments)
            ax.set_xlim(-5, len(prot) + 5)
            ax.hlines([0], 0, len(prot), linestyles="dotted", color="black", alpha=0.4)
            ax.set_title("Dissimilarity score per segment")
            midpoints = [s.m_idx for s in sc_segments]
            scores = np.abs(segment_scores.logprobs_s)
            max_score = np.max(scores)
            ax.plot(midpoints, scores , 'o',
                    alpha=0.5, color=epstyle.color_prob_s, label='Shift')
            scores = np.abs(segment_scores.logprobs_e)
            max_score = max(max_score, np.max(scores))
            ax.plot(midpoints, scores, 'o',
                    alpha=0.5, color=epstyle.color_prob_e, label='Compression')

            x, y = [], []
            path = segment_scores.pair.segments_to_path()
            for score, point in zip(segment_scores.logprobs_a, path):
                x.append(point[0])
                y.append(score)
            scores = np.abs(y)
            max_score = max(max_score, np.max(scores))
            ax.scatter(x, scores,
                       alpha=0.5, color=epstyle.color_prob_a, s=2, label='Amplitude')
            ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.1, 0.3))
            ax.set_ylim(-1.0, max(1.0, max_score + 0.2))
            ax.set_xticks([])
            for x in [segment.s_idx for segment in sc_segments] + [sc_segments[-1].e_idx]:
                ax.axvline(
                    x,
                    linestyle="dotted",
                    color=color_shade,
                    alpha=0.4,
                )

            # Dynamic Subsequence Warping
            ax0 = fig.add_subplot(gs[gs_height_blocks+2, 0])
            axs.append(ax0)
            ax1 = fig.add_subplot(gs[gs_height_blocks+3, 0])
            axs.append(ax1)
            ax0.set_xlim(-5, len(prot) + 5)
            ax1.set_xlim(-5, len(prot) + 5)
            ax0.set_xticks([])

            # plot_warping needs a fixed layout to draw the lines between axes
            # fig.tight_layout()
            # fig.canvas.draw()

            ax0.plot(segment_scores.pair.series_to, color=color_series_to,
                     linestyle='dotted')
            segment_scores.pair.plot_warping(
                axs=[ax0, ax1], fig=fig,
                nice_layout=False, show_legend=False,
            )

        if rel_thr != 1:
            fig.suptitle(f'Settings: rel_thr={rel_thr}',
                        x=0.03, y=0.03, ha='left', fontsize=9)

        if filename is not None:
            fig.savefig(str(filename))
            plt.close(fig)
            fig = None
            axs = None
        return fig, axs


class ExponentialInterval(InnerDistBaseObject):
    """Euclidean interval.
    This captures both the shift in value and the expansion of uncertainty.
    """

    def __init__(self, rel_thr=1):
        self.rel_thr = rel_thr

    def inner_dist(self, x, y):
        """x are the prototype values, y the series to be tested.
        """
        x_v, x_l, x_r = x
        diff = y - x_v
        scale = x_l if diff <= 0 else x_r
        diff = abs(diff)
        # normality_logprob
        lcdf = np.minimum(0.0, self.rel_thr - np.divide(diff, scale))
        # make abs to have a distance
        return abs(lcdf)

    def inner_dists(self, xs, ys):
        diffs = ys - xs[:, 0]
        scales = np.multiply(
            diffs <= 0, xs[:, 1]
        ) + np.multiply(diffs > 0, xs[:, 2])
        diffs = abs(diffs)
        # normality_logprob
        lcdf = np.minimum(0.0, self.rel_thr - np.divide(diffs, scales))
        # make abs to have a distance
        return np.abs(lcdf)

    def result(self, x):
        return x

    def inner_val(self, x):
        return x


