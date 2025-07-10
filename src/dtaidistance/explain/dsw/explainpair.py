# -*- coding: UTF-8 -*-
"""
dtaidistance.explain.dws.explainpair
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(requires version 2.4.0 or higher)

Explain the warping path.

Usage

::

    ep = ExplainPair(ya, yb, delta_rel=2, delta_abs =0, approx_prune=True)
    print(ep.distance_approx())
    ep.plot_warping("/path/to/figure.png")


:copyright: Copyright 2025 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import functools
import heapq
import sys
from bisect import bisect_left, bisect_right
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ... import dtw_visualisation as dtwvis
from ... import innerdistance
from ...dtw import warping_path, DTWSettings, warping_path_fast, warping_paths

diag_angle = np.pi / 4

color_series = "#2E81B9"
color_shade = "#F57A00"
color_shade_dark = "#5F3E1C"


class ApproxType(str, Enum):
    MAX_INDEX = "max_index"
    MEAN_INDEX = "mean_index"  # Not implemented yet
    MAX_FACTOR = "max_factor"
    MAX_DIFF = "max_diff"
    MAX_FACTOR_AND_DIFF = "max_factor_and_diff"
    MAX_DIST = "max_dist"
    MAX_FACTOR_LOOSE = "max_factor_loose"

    def to_int(self):
        return list(ApproxType).index(self)

    @staticmethod
    def from_int(value):
        return list(ApproxType)[value]


class SplitStrategy(str, Enum):
    SPATIAL_DIST = "spatialdist"
    PATH_DIFF = "pathdiff"  # TODO: depricate
    DERIV = "deriv"
    DERIV_DIST = "derivdist"

    def to_int(self):
        return list(SplitStrategy).index(self)

    @staticmethod
    def from_int(value):
        return list(SplitStrategy)[value]


@dataclass
class Segment:
    s_idx: int
    """Start index in series"""
    e_idx: int
    """End index in series"""
    angle: float
    """Angle"""
    shift: int
    """Shift left (neg) or right (pos)"""
    elasticity: int
    """Expansion (pos) or Compression (neg)"""

    def __iter__(self):
        return iter([self.s_idx, self.e_idx, self.angle, self.shift, self.elasticity])

    def copy(self):
        return Segment(self.s_idx, self.e_idx, self.angle, self.shift, self.elasticity)

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


@dataclass
class PathSegment:
    s_idx_p: int
    """Start index in path"""
    e_idx_p: int
    """End index in path"""
    s_idx_y: int
    """Start index in other series"""
    e_idx_y: int
    """End index in other series"""
    segment: Segment

    def compression_logratio(self):
        return np.log(self.e_idx_y - self.s_idx_y + 1) - np.log(self.e_idx - self.s_idx + 1)

    def compression_ratio(self):
        return (self.e_idx_y - self.s_idx_y + 1) / (self.e_idx - self.s_idx + 1)

    def length(self):
        return self.segment.length()

    def length_y(self):
        return self.e_idx_y - self.s_idx_y + 1

    @property
    def s_idx(self):
        return self.segment.s_idx

    @s_idx.setter
    def s_idx(self, s_idx):
        self.segment.s_idx = s_idx

    @property
    def e_idx(self):
        return self.segment.e_idx

    @e_idx.setter
    def e_idx(self, e_idx):
        self.segment.e_idx = e_idx

    @property
    def angle(self):
        return self.segment.angle

    @property
    def shift(self):
        return self.segment.shift

    @shift.setter
    def shift(self, value):
        self.segment.shift = value

    @property
    def elasticity(self):
        return self.segment.elasticity

    @property
    def expansion(self):
        return self.segment.expansion

    @property
    def compression(self):
        return self.segment.compression

    def __iter__(self):
        return iter(
            [
                self.s_idx_p,
                self.e_idx_p,
                self.s_idx,
                self.e_idx,
                self.angle,
                self.shift,
                self.elasticity,
            ]
        )

    def segment_copy(self):
        return self.segment.copy()

    def copy(self):
        return PathSegment(
            self.s_idx_p, self.e_idx_p, self.s_idx_y, self.e_idx_y, self.segment.copy()
        )


class ApproxSettings:
    def __init__(self, approx_type=ApproxType.MAX_FACTOR_AND_DIFF,
                 delta_rel=1, delta_abs=0.1,
                 approx_prune=True, approx_local=True,
                 split_strategy=SplitStrategy.SPATIAL_DIST):
        """
        Approximation settings for ExplainPair. See more information in the ExplainPair class.
        """
        if not isinstance(approx_type, ApproxType):
            if type(approx_type) is int:
                approx_type = ApproxType.from_int(approx_type)
            elif type(approx_type) is str:
                approx_type = ApproxType(approx_type)
            else:
                raise ValueError(f'Unknown type for approx_type')
        self.approx_type = approx_type
        if delta_abs is None:
            raise ValueError("delta_abs must be provided")
        if self.approx_type == ApproxType.MAX_FACTOR_AND_DIFF and delta_rel is None:
            raise ValueError("delta_rel must be provided for MAX_FACTOR_AND_DIFF")
        self.delta_rel = delta_rel
        self.delta_abs = delta_abs
        self.approx_prune = approx_prune
        self.approx_local = approx_local
        if not isinstance(split_strategy, SplitStrategy):
            if type(split_strategy) is int:
                split_strategy = SplitStrategy.from_int(split_strategy)
            elif type(split_strategy) is str:
                split_strategy = SplitStrategy(split_strategy)
            else:
                raise ValueError(f'Unknown type for split_strategy')
        self.split_strategy = split_strategy

    @staticmethod
    def wrap(s=None, **kwargs):
        if s is None:
            return ApproxSettings(**kwargs)
        if isinstance(s, ApproxSettings):
            return s
        raise ValueError(f'Unknown argument type for ApproxSettings: {s}')

    def __repr__(self):
        return (
            f"ApproxSettings(approx_type={self.approx_type}, "
            f"delta_rel={self.delta_rel}, delta_abs={self.delta_abs}, "
            f"approx_prune={self.approx_prune}, approx_local={self.approx_local}, "
            f"split_strategy={self.split_strategy})"
        )

    def kwargs(self):
        return {
            "approx_type": self.approx_type,
            "delta_rel": self.delta_rel,
            "delta_abs": self.delta_abs,
            "approx_prune": self.approx_prune,
            "approx_local": self.approx_local,
            "split_strategy": self.split_strategy
        }

    def c_kwargs(self):
        return {
            "approx_type": self.approx_type.to_int(),
            "delta_rel": self.delta_rel,
            "delta_abs": self.delta_abs,
            "approx_prune": self.approx_prune,
            "approx_local": self.approx_local,
            "split_strategy": self.split_strategy.to_int()
        }

    def to_h5_group(self, group):
        for key, value in self.c_kwargs().items():
            group.attrs[key] = value

    @staticmethod
    def from_h5_group(group):
        kwargs = {}
        for attr in ["approx_type", "delta_rel", "delta_abs", "approx_prune",
                     "approx_local", "split_strategy"]:
            if attr in group.attrs:
                kwargs[attr] = group.attrs[attr]
        return ApproxSettings(**kwargs)


class ExplainPair:
    def __init__(
            self,
            series_from,
            series_to,
            approx_type=ApproxType.MAX_FACTOR_AND_DIFF,
            delta_rel=1,
            delta_abs=None,
            approx_prune=True,
            approx_local=True,
            split_strategy=SplitStrategy.SPATIAL_DIST,
            onlychanges=None,
            path=None,
            dtw_settings=None,
            save_intermediates=False
    ):
        """Compute segments and variations that explain the warping path
        between two series.

        :param series_from: Series from
        :param series_to: Series to
        :param approx_type: Type of approximation to use.

            Ensures that the new DTW distance after the approximation is within
            a bound. Let d' be the DTW distance of the new path, and d be the
            DTW distance of the original path. The possible choices are:

            * ``max_index``: Absolute position based.
                Allow to deviate from the original path by at most delta_abs positions.
                Not related to the d' and d.
                It is the maximum allowed spatial distance between each new subpath and the corresponding original subpath.
            * ``max_factor``: Relative distance based.
                :math:`d' \\leq d * (1 + \\delta_{rel})`
            * ``max_factor_loose``: Relative distance based but looser.
                This allows also simplifying subsequences with a very
                low distance. Thus a good match with a distance close to zero
                and where the simplification would lead to a distance a bit
                higher than zero.

                :math:`d' \\leq d * (1 + 1.1*\\delta_{rel})`
            * ``max_diff``: Absolute distance based:
                :math:`d' \\leq d + \\delta_{abs}`
            * ``max_factor_and_diff``: Combined distance based.
                :math:`d' \\leq d * (1 + \\delta_{rel}) + \\delta_{abs}`
            * ``max_dist``: Absolute distance based
                :math:`d' <= \\delta_{abs}`

        :param delta_rel: User-defined relative tolerance parameter.
            It controls how much deviation is allowed based on the original DTW distance.
            It allows flexibility proportional to the distance of the original path.
        :param delta_abs: User-defined absolute tolerance parameter.
            It sets a fixed allowance for deviation.
            It allows flexibility regardless of the distance of the original path.
            It has different meanings depending on the approx_type.
        :param approx_prune: Whether to add a last round that merges segments bottom-up.
        :param approx_local: Whether to apply tolerance criterion check locally or globally during the pruning phase.
            If True, apply the local tolerance criterion to check the cost of each segment;
            if False, apply the global tolerance criterion to check the cost of the full sequence matching.
        :param split_strategy: The strategy to use for deciding the splitting points:

            * ``spatialdist``: Split on the point on the path the furthest
                away from the straight path.
            * ``deriv``: Split on the point on the path that has the highest
                local second derivative.
            * ``derivdist``: Split on the point on the path that has the largest
                difference in cost between the point on the path and the
                closest point on the straight path. Approximated with the
                locally computed first and second derivative around the point
                on the path.

        :param onlychanges: Only return segments with changes above this threshold
        :param path: Use given warping path
        :param dtw_settings: Object of type :class:`DTWSettings`
            This variable is ignored if `path` is given.
        :param save_intermediates: Save intermediate results
        """
        self.series_from = series_from
        self.series_to = series_to
        self.approx_type = approx_type
        self.delta_rel = delta_rel
        self.delta_abs = delta_abs
        if self.delta_abs is None:
            if self.approx_type == ApproxType.MAX_FACTOR_LOOSE:
                self.delta_abs = 0.1  # will be used to set delta_abs to delta_abs*delta_rel*d
            else:
                self.delta_abs = 0.1
        self.approx_prune = approx_prune
        self.approx_local = approx_local
        self.split_strategy = split_strategy
        self.comb_op = max  # max or min
        self.onlychanges = onlychanges
        self.save_intermediates = save_intermediates
        self.segments = None
        self._variations = None
        self.intermediates = None

        self.dtw_settings = DTWSettings.wrap(dtw_settings)
        if self.dtw_settings.use_c is False:
            self.path = (
                warping_path(self.series_from, self.series_to, **self.dtw_settings.kwargs())
                if path is None
                else path
            )
        else:
            self.path = (
                warping_path_fast(
                    self.series_from, self.series_to, **self.dtw_settings.kwargs()
                )
                if path is None
                else path
            )
        self.segments, self.line2 = self.path_to_segments(self.path, onlychanges=onlychanges)

        # Amplitude variations (computed lazily)
        self._variations = None

    def path_to_segments(self, path, onlychanges=None):
        """Compute segments and variations that explain the warping path between two series.

        :param path: Warping path
        :param onlychanges: Only return segments with changes above this threshold
        :return: (segments, variations)
        """
        # Shift and elasticity
        line = np.array(path)
        # line2, lidxs = rdp(line, epsilon=epsilon)
        if self.approx_type == "max_index":
            line2, lidxs = rdp_vectorized(line, epsilon=self.delta_abs)
        else:
            line2, lidxs = self.rdp_ssm(line)
        # 0 is maximal compression
        # acomp = np.pi / 5  # Compression
        # adiag = np.pi / 4  # Shift, angle diagonal
        # aexpa = np.pi / 3  # Expansion  (7/24)
        # pi/2 is maximal expansion

        segments = []
        for idx in range(len(lidxs) - 1):
            bp = line2[idx]  # type: np.ndarray[int]
            ep = line2[idx + 1]  # type: np.ndarray[int]
            dx = ep[0] - bp[0]
            dy = ep[1] - bp[1]
            if dx == 0:
                a = np.pi / 2
            else:
                a = np.arctan(dy / dx)
            # print(f"{bp}-{ep} : {a:.3f}")
            # Shift based on middle point of segment
            # shift = round((bp[1] + ep[1]) / 2 - (bp[0] + ep[0]) / 2)
            # Shift based on point closest to diagonal
            denom = bp[1] - ep[1] - bp[0] + ep[0]
            if np.isclose(denom, 0.0):
                t = 0  # parallel, all points are equal
            else:
                t = max(0.0, min(1.0, (bp[1] - bp[0]) / denom))
            shift = round(bp[1] + t * (ep[1] - bp[1]) - bp[0] - t * (ep[0] - bp[0]))
            expansion = dy - dx
            # if a < acomp:
            #     # Compression
            #     compression = dx - dy
            # elif a < aexpa:
            #     # Shift
            #     shift = round((bp[1] + ep[1])/2 - (bp[0] + ep[0])/2)
            # else:
            #     # Expansion
            #     expansion = dy - dx
            if (
                    onlychanges is None
                    or abs(shift) >= onlychanges
                    or abs(expansion) >= onlychanges
            ):
                segment = Segment(bp[0], ep[0], a, shift, expansion)
                segment = PathSegment(lidxs[idx], lidxs[idx + 1], bp[1], ep[1], segment)
                segments.append(segment)

        return segments, line2

    def rdp_ssm(self, points):
        """Ramer-Douglas-Peucker algorithm to simplify a path in a
        Self-Similarity Matrix (SSM).

        Instead of spatial distance to simplifying line, use difference in
        cumulative cost along path. Simplification is allowed based
        on an ub_m or difference on top of the current dtw distance.

        The type of relaxation is dependent on self.approx_type

        :param points: 2D numpy array (points x dimensions)
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        queue = deque([(0, len(points) - 1)])
        result = set()

        # Compute cumulative cost over path
        inner_dist, inner_res, inner_val = innerdistance.inner_dist_fns(self.dtw_settings.inner_dist)
        ccostv_o = np.empty(len(points))  # Cumulative cost vector for original path
        ccost_o = 0  # Cumulative cost
        for idx in range(len(points)):
            i, j = points[idx]
            ccost_o += inner_dist(self.series_from[i], self.series_to[j])
            ccostv_o[idx] = ccost_o
        lenr_o = len(points)  # Length of original path remaining

        tolerance_factor_rel, tolerance_factor_ab = self.compute_tolerance_criterion_factors(ccost_o, lenr_o, inner_res,
                                                                                             inner_val)

        # Split selection
        if self.split_strategy == SplitStrategy.SPATIAL_DIST:
            split_selection = self.max_deviation_from_line
        elif self.split_strategy == SplitStrategy.PATH_DIFF:
            split_selection = functools.partial(self.max_change_in_path, inner_dist)
        elif self.split_strategy == SplitStrategy.DERIV:
            # 2nd derivative is computed upfront
            split_selection = functools.partial(self.max_2ndderiv_in_path,
                                                self.get_2ndderiv_in_path(inner_dist, points))
        elif self.split_strategy == SplitStrategy.DERIV_DIST:
            split_selection = functools.partial(self.max_2ndderiv_deviation,
                                                self.get_1stderiv_in_path(inner_dist, points),
                                                self.get_2ndderiv_in_path(inner_dist, points))
        else:
            raise AttributeError(f"Unknown split strategy: {self.split_strategy}")

        handled_segments = []
        # Each time for a given segemnt, either it is accepted, or a new splitting point within the segment is created if the segment is not accepted.
        while len(queue) > 0:
            i0, i1 = queue.popleft()

            if i1 - i0 <= 1:
                # Nothing to simplify
                result.add(i0)
                result.add(i1)
                continue
            p0, p1 = points[i0], points[i1]
            ccostp_o = ccostv_o[i1] - ccostv_o[i0]  # Cost of partial original path
            lenp_o = i1 - i0

            ccostp_a, lenp_a = self._line_cost(p0, p1, inner_dist,
                                               include_begin=False, include_end=True)

            # print(f"== ({i0},{i1}) / ({p0},{p1}) ==")
            # print(f"{ccostp_a:.4f} <? max({ub_a:.4f}*{lenp_o},{ccostp_o:.4f}*{ub_m:.2f})"
            #       f" = {max(lenp_o*ub_a, ccostp_o*ub_m):.4f}")

            # Margin is a percentage of allowed increase of the dtw
            # Using a percentage allows to apply this to all segments,
            # irrespective of their length, and offers some guarantees
            # on the total deviation.
            # It exists out of two bounds, of which the largest is taken.
            #  ccostp_o * (1 + tolerance_factor_rel): Is the new path only a bit worse than the original path.
            #     This part is useful when a part of the path has a large distance. In
            #     that part of the path it is ok to allow for a larger deviation.
            # ccostp_o + lenp_o * tolerance_factor_ab,: Is the new path still better than the expected cost
            #     averaged over the entire series. This part is useful when a part of
            #     the path is a near perfect match and multiplying a very small number
            #     with the 'tolerance_factor_rel' is still a very small number.
            if ccostp_a <= self.comb_op(ccostp_o + lenp_o * tolerance_factor_ab, ccostp_o * (1 + tolerance_factor_rel)):
                result.add(i0)
                result.add(i1)
                if self.save_intermediates:
                    handled_segments.append([i0, i1, None])  # Simplify the path between the current two points
            else:
                # Retry with current largest deviation from the straight
                # line as point in between (like in the original rdp algorithm)
                _, idxmax = split_selection(points, i0, i1)
                if idxmax == i0:
                    idxmax = i0 + 1
                queue.append((i0, idxmax))
                queue.append((idxmax, i1))

                if self.save_intermediates:
                    handled_segments.append([i0, i1, idxmax])  # Split the path between the current two points

        if self.approx_prune:
            result = self.remove_segments(points, result, ccostv_o)

        result = sorted(result)
        new_points = np.zeros((len(result), 2), dtype=int)
        for idx, ridx in enumerate(result):
            new_points[idx] = points[ridx]
        if self.save_intermediates:
            self.intermediates = handled_segments
        return new_points, result

    def remove_segments(self, points, idxs, ccostv_o):
        """Remove segments as longs as the cost still satisfies the tolerance criterion.

        Compared to rdp_ssm, which is top-down, this method is bottom-up. But therefore
        also slower. Therefore, first rdp_ssm is used.

        :param points:  2D numpy array (points x dimensions)
        :param idxs: the idxs to removed, only the retained idxs after the top-down rdp_ssm are passed
        :param ccostv_o: Cumulative cost vector for original path
        """
        inner_dist, cost2dist, dist2cost = innerdistance.inner_dist_fns(self.dtw_settings.inner_dist)
        ccost_o = ccostv_o[-1]
        global_upperbound = self.compute_bounds_global(ccost_o, cost2dist, dist2cost)
        queue = []
        new_idxs = SortedList(idxs)
        cnt = np.zeros(1)
        tolerance_factor_rel, tolerance_factor_ab = self.compute_tolerance_criterion_factors(ccost_o, len(points),
                                                                                             cost2dist, dist2cost)

        @functools.cache
        def line_cost(p0, p1):
            cnt[0] += 1
            ccosti, _lena = self._line_cost(p0, p1, inner_dist, include_begin=False, include_end=True)
            return ccosti

        ccost_a = 0
        for i0, i1 in zip(new_idxs, new_idxs[1:]):
            p0, p1 = points[i0], points[i1]
            ccost_a += line_cost((p0[0], p0[1]), (p1[0], p1[1]))
        ccost_a += inner_dist(self.series_from[-1], self.series_to[-1])
        # print(f'{ccost_a=:.4f}, {upperbound=}')

        for i0, i1, i2 in zip(new_idxs, new_idxs[1:], new_idxs[2:]):
            heapq.heappush(queue, (min(i2 - i1, i1 - i0), (i0, i1, i2)))

        while len(queue) > 0:
            _, (i0, i1, i2) = heapq.heappop(queue)
            # print(f'== {i0},{i1},{i2}')
            try:
                new_idxs.index(i0)
                new_idxs.index(i1)
                new_idxs.index(i2)
            except ValueError:
                # Already removed
                continue
            p0, p1, p2 = points[i0], points[i1], points[i2]
            # print(p0, p1, p2)

            c_02a = line_cost((p0[0], p0[1]), (p2[0], p2[1]))

            if self.approx_local:
                ccostp_o = ccostv_o[i2] - ccostv_o[i0]  # Cost of partial original path
                lenp_o = i2 - i0
                do_simplify = (c_02a <= self.comb_op(ccostp_o + lenp_o * tolerance_factor_ab,
                                                     ccostp_o * (tolerance_factor_rel + 1)))
            else:
                c_01 = line_cost((p0[0], p0[1]), (p1[0], p1[1]))
                c_12 = line_cost((p1[0], p1[1]), (p2[0], p2[1]))
                c_02 = c_01 + c_12
                # print(f'{ccost_a=:.4f} - {c_02=:.4f} + {c_02a=:.4f} < {upperbound=:.4f}')
                if ccost_a - c_02 + c_02a < global_upperbound:
                    do_simplify = True
                    ccost_a = ccost_a - c_02 + c_02a
                else:
                    do_simplify = False

            if do_simplify:
                # print(f'Remove {i1}')
                try:
                    i_n = new_idxs.find_lt(i0)
                    i0n, i1n, i2n = i_n, i0, i2
                    heapq.heappush(queue, (min(i1n - i0n, i2n - i1n), (i0n, i1n, i2n)))
                    # print(f'Add {i0n}, {i1n}, {i2n}')
                except ValueError:
                    pass
                try:
                    i_n = new_idxs.find_gt(i2)
                    i0n, i1n, i2n = i0, i2, i_n
                    heapq.heappush(queue, (min(i1n - i0n, i2n - i1n), (i0n, i1n, i2n)))
                    # print(f'Add {i0n}, {i1n}, {i2n}')
                except ValueError:
                    pass
                new_idxs.remove(i1)
            else:
                pass

        # print(f'remove_segments: {len(idxs)} -> {cnt[0]=}')
        return new_idxs

    def compute_bounds_global(self, ccostc, cost2dist, dist2cost):
        """Compute the bound on the cost of the global approximated path.

        :param ccostc: Accumulated cost
        :param cost2dist: Transform cost to distance
        :param dist2cost: Transform distance to cost
        """
        ccost = cost2dist(ccostc)
        if self.approx_type == ApproxType.MAX_FACTOR:
            ub = (self.delta_rel + 1) * ccost
        elif self.approx_type == ApproxType.MAX_FACTOR_LOOSE:
            ub = ((1+self.delta_abs)*self.delta_rel + 1) * ccost
        elif self.approx_type == ApproxType.MAX_FACTOR_AND_DIFF:
            ub = (1 + self.delta_rel) * ccost +  self.delta_abs
        elif self.approx_type == ApproxType.MAX_DIFF:
            ub = self.delta_abs + ccost
        elif self.approx_type == ApproxType.MAX_DIST or self.approx_type == ApproxType.MAX_INDEX:
            ub = self.delta_abs #todo check
        else:
            raise ValueError(f'Unknown approximation type: {self.approx_type}')
        ub = dist2cost(ub)
        return ub

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
        if self.approx_type == ApproxType.MAX_FACTOR:
            ccost_ub = cost2dist(ccost) * (self.delta_rel)
            try:
                ub_m = ((dist2cost(ccost_ub)) /
                        (ccost))
            except (ValueError, ZeroDivisionError):
                ub_m = 0
            ub_a = 0
        elif self.approx_type == ApproxType.MAX_FACTOR_LOOSE:
            ccost_ub = cost2dist(ccost) * (self.delta_rel)
            try:
                ub_m = ((dist2cost(ccost_ub)) /
                        (ccost))
            except (ValueError, ZeroDivisionError):
                ub_m = 0
            ub_a = (dist2cost(cost2dist(ccost)*(1 + self.delta_rel*self.delta_abs)) - ccost) / length
        elif self.approx_type == ApproxType.MAX_FACTOR_AND_DIFF:
            ccost_ub = cost2dist(ccost) * (self.delta_rel)
            try:
                ub_m = ((dist2cost(ccost_ub)) /
                        (ccost))
            except (ValueError, ZeroDivisionError):
                ub_m = 0
            ub_a = (dist2cost(cost2dist(ccost) + self.delta_abs) - ccost) / length
            # retain history for reference
            # ccost_ub = cost2dist(ccost) * (1 + self.epsilon)
            # try:
            #     ub_m = ((dist2cost(ccost_ub) - ccostpa) /
            #             (ccost - ccostp))
            # except (ValueError, ZeroDivisionError):
            #     ub_m = 0
            # # ub_a = max((min(self.epsilon, 1) * (ccost - ccostp)) / lengthr,
            # #            np.finfo(self.series_from.dtype).eps)
            # ub_a = max((min(self.epsilon, 1) * (ccost - ccostp) - ccost) / lengthr,
            #            np.finfo(self.series_from.dtype).eps)

        elif self.approx_type == ApproxType.MAX_DIFF:
            # ub_m = self.epsilon / ccost + 1
            # ccost_ub = self.epsilon + ccost
            ub_m = 0
            # ub_a = (dist2cost(cost2dist(ccost)  + self.epsilon) - ccostpa) / lengthr
            ub_a = (dist2cost(cost2dist(ccost) + self.delta_abs) - ccost) / length
        elif self.approx_type == ApproxType.MAX_DIST or self.approx_type == ApproxType.MAX_INDEX:
            # ub_m = self.epsilon / ccost
            # ccost_ub = self.epsilon
            ub_m = 0
            # ub_a = (dist2cost(self.epsilon) - ccostpa) / lengthr
            ub_a = (dist2cost(self.delta_abs) - ccost) / length
        else:
            raise ValueError(f'Unknown approximation type: {self.approx_type}')

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

    def get_1stderiv_in_path(self, inner_dist, points, h=1):
        """Compute the first derivative of each point in the cost
        matrix (or self-similarity matrix)."""
        ders = np.zeros(len(points))
        i_of_m = len(self.series_from) - h - 1
        i_ot_m = len(self.series_to) - h - 1
        for idx in range(0, len(points)):
            i_of, i_ot = points[idx]
            if i_of < h or i_of > i_of_m or i_ot < h or i_ot > i_ot_m:
                # Compute derivatives close to the border
                der = (abs((inner_dist(self.series_from[max(0, i_of - h)], self.series_to[max(0, i_ot - h)]) -
                            inner_dist(self.series_from[min(i_of_m, i_of + h)], self.series_to[min(i_ot_m, i_ot + h)]))
                           / (2*h)) +
                       abs((inner_dist(self.series_from[min(i_of_m, i_of + h)], self.series_to[max(0, i_ot - h)]) -
                           inner_dist(self.series_from[max(0, i_of - h)], self.series_to[min(i_ot_m, i_ot + h)]))
                           / (2*h))
                       ) / 2
            else:
                # Centered difference (1st order approx) in two (diagonal)
                # directions and averaged
                der = (abs((inner_dist(self.series_from[i_of-h], self.series_to[i_ot-h]) -
                            inner_dist(self.series_from[i_of+h], self.series_to[i_ot+h])
                            ) / (2*h)) +
                       abs((inner_dist(self.series_from[i_of+h], self.series_to[i_ot-h]) -
                            inner_dist(self.series_from[i_of-h], self.series_to[i_ot+h])
                            ) / (2*h))
                       ) / 2
            ders[idx] = abs(der)
        # If the first derivative is zero, then distance has little impact
        # in the max_2ndderiv_deviation approach
        min_ders = np.max(ders) * 0.01
        ders[ders < min_ders] = min_ders
        return ders

    def get_2ndderiv_in_path(self, inner_dist, points, h=1):
        """Compute the second derivative of each point in the cost
        matrix (or self-similarity matrix).

        This method ignores the direction and computes the average
        absolute value of the derivative in the diagonal directions.
        The goal is to select points that have a rapidly changing point
        in the cost matrix.

        The centered difference approximations (along the diagonals)
        method is used.
        """
        ders = np.zeros(len(points))
        i_of_m = len(self.series_from) - h - 1
        i_ot_m = len(self.series_to) - h - 1
        for idx in range(0, len(points)):
            i_of, i_ot = points[idx]
            if i_of < h or i_of > i_of_m or i_ot < h or i_ot > i_ot_m:
                # Compute derivatives close to the border
                der = (abs(inner_dist(self.series_from[max(0, i_of - h)], self.series_to[max(0, i_ot - h)])+
                           inner_dist(self.series_from[min(i_of_m, i_of + h)], self.series_to[min(i_ot_m, i_ot + h)])-
                           2 * inner_dist(self.series_from[i_of], self.series_to[i_ot])
                           ) / (h**2) +
                       abs(inner_dist(self.series_from[min(i_of_m, i_of + h)], self.series_to[max(0, i_ot - h)])+
                           inner_dist(self.series_from[max(0, i_of - h)], self.series_to[min(i_ot_m, i_ot + h)])-
                           2*inner_dist(self.series_from[i_of], self.series_to[i_ot])
                           ) / (h**2)
                       ) / 2
            else:
                # Centered difference approximations (along the diagonals)
                # Could also have been five-point stencil 2nd derivative?
                der = (abs(inner_dist(self.series_from[i_of-h], self.series_to[i_ot-h])+
                           inner_dist(self.series_from[i_of+h], self.series_to[i_ot+h])-
                           2 * inner_dist(self.series_from[i_of], self.series_to[i_ot])
                           ) / (h**2) +
                       abs(inner_dist(self.series_from[i_of+h], self.series_to[i_ot-h])+
                           inner_dist(self.series_from[i_of-h], self.series_to[i_ot+h])-
                           2*inner_dist(self.series_from[i_of], self.series_to[i_ot])
                           ) / (h**2)
                       ) / 2
            ders[idx] = abs(der)
        return ders

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

    def max_2ndderiv_deviation(self, ders1, ders2, points, i0, i1):
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
        p0, p1 = points[i0], points[i1]
        p0p1norm = np.linalg.norm(p1 - p0)
        p0p1normsqr = p0p1norm ** 2
        distmax = 0
        idxmax = i0
        for idx in range(i0, i1):
            der1 = ders1[idx]
            der2 = ders2[idx]
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
            # Second-order Taylor expansion to approximate the difference
            # in the cost matrix based on the local derivatives
            dist2 = der1*dist + 1/2*der2*dist**2
            dist = dist2

            if dist > distmax:
                distmax = dist
                idxmax = idx
        return distmax, idxmax

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

    def distance(self, per_segment=False):
        dist = 0
        inner_dist, inner_res, _inner_val = innerdistance.inner_dist_fns(self.dtw_settings.inner_dist)
        dists = [] if per_segment else None

        for segment in self.segments:  # type: PathSegment
            dist_seg = 0
            for i_p in range(segment.s_idx_p, segment.e_idx_p):
                point = self.path[i_p]
                dist_seg += inner_dist(self.series_from[point[0]], self.series_to[point[1]])
            dist += dist_seg
            if per_segment:
                dists.append(dist_seg)
        point = self.path[self.segments[-1].e_idx_p]
        idist = inner_dist(self.series_from[point[0]], self.series_to[point[1]])
        if per_segment:
            dists.append(idist)
        dist += idist
        dist = inner_res(dist)
        if per_segment:
            return dist, dists
        return dist

    def distance_approx(self, per_segment=False):
        """DTW Distance for approximated path."""
        dist = 0
        inner_dist, inner_res, _inner_val = innerdistance.inner_dist_fns(self.dtw_settings.inner_dist)
        dists = [] if per_segment else None

        for segment in self.segments:  # type: PathSegment
            # Bresenham's line algorithm
            i_f, i_fe = segment.s_idx, segment.e_idx
            i_t, i_te = segment.s_idx_y, segment.e_idx_y
            d_f = i_fe - i_f
            d_t = i_t - i_te
            error = d_f + d_t

            dist_seg = 0
            while True:
                if i_f == i_fe and i_t == i_te:
                    # Stop and skip inner_dist for last point (overlaps with next segment)
                    break
                dist_seg += inner_dist(self.series_from[i_f], self.series_to[i_t])
                e2 = 2 * error
                if e2 >= d_t:
                    error += d_t
                    i_f += 1
                if e2 <= d_f:
                    error += d_f
                    i_t += 1
            dist += dist_seg
            if per_segment:
                dists.append(dist_seg)
        i_f, i_t = self.segments[-1].e_idx, self.segments[-1].e_idx_y
        idist = inner_dist(self.series_from[i_f], self.series_to[i_t])
        if per_segment:
            dists.append(idist)
        dist += idist
        dist = inner_res(dist)
        if per_segment:
            return dist, dists
        return dist

    def dsw_path(self):
        """The piece-wise linearized path."""
        return self.segments_to_path()

    def segments_to_path(self):
        """DTW Distance for approximated path."""
        path = []
        for segment in self.segments:  # type: PathSegment
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
        path.append((self.segments[-1].e_idx, self.segments[-1].e_idx_y))
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
        variations = self.get_variations(on_segments=False)
        self._variations = variations
        return self._variations

    def get_variations(self, on_segments=False):
        """Compute the amplitude variations

        :param on_segments: Compute the variations based on the linear segments
            instead of the original optimal path
        :return:
        """
        if on_segments:
            path = self.segments_to_path()
        else:
            path = self.path
        variations = np.zeros((len(self.series_from), 2))
        tvalues = defaultdict(lambda: ([], []))
        for fi, ti in path:
            v = self.series_to[ti] - self.series_from[fi]
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

    def plot(self, filename=None, fig=None):
        fig, _ = plot_explain(
            self.series_from,
            self.segments,
            self.variations,
            filename=filename,
            fig=fig,
        )
        return fig

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
            show_elasticity=True,
            color_elasticity=False,
            alt_vis=False,
    ):
        if show_amplitude:
            if alt_vis:
                variations = self.get_bounds(on_segments=True)
            else:
                variations = self.get_variations(on_segments=True)
        else:
            variations = None
        if alt_vis:
            fn = plot_warping2
        else:
            fn = plot_warping
        return fn(
            self.series_from,
            self.series_to,
            self.segments,
            self.path,
            filename=filename,
            fig=fig,
            axs=axs,
            series_line_options=series_line_options,
            warping_line_options=warping_line_options,
            elasticity_line_options=elasticity_line_options,
            show_xticks=show_xticks,
            show_yticks=show_yticks,
            tick_kwargs=tick_kwargs,
            variations=variations,
            show_elasticity=show_elasticity,
            color_elasticity=color_elasticity,
        )

    def plot_explanation_and_warping(self, filename=None, relsize=0.4):
        if self.line2[-1][0] != self.line2[-1][1]:
            raise AttributeError(f"Not supported for two series of different lengths "
                                 f"({self.line2[-1][0]} != {self.line2[-1][1]})")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        fig, gs = plot_explain(
            self.series_from, self.segments, self.variations, fig=fig
        )
        gs.update(top=0.95, bottom=relsize + 0.05)
        new_gs = fig.add_gridspec(2, 1, top=relsize - 0.05, bottom=0.05)
        axs = [fig.add_subplot(new_gs[0, 0]), fig.add_subplot(new_gs[1, 0])]
        axs[0].set_xlim(-5, len(self.series_from) + 5)
        axs[0].set_title("Warped from time series")
        axs[1].set_xlim(-5, len(self.series_from) + 5)
        axs[1].set_title("Warped to time series")
        self.plot_warping(fig=fig, axs=axs)
        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
            fig = None
        return fig

    def plot_segments(self, filename=None, show_values=False, path_kwargs=None, matshow_kwargs=None, showdist=True,
                      tick_kwargs=None, dsw_path_kwargs=None):
        import matplotlib.pyplot as plt
        ya, yb = self.series_from, self.series_to
        path = self.line2
        dist, paths = warping_paths(ya, yb)
        dist_approx = self.distance_approx()
        fig, axs = dtwvis.plot_warpingpaths(ya, yb, paths, path=self.path, path_kwargs=path_kwargs,
                                            matshow_kwargs=matshow_kwargs, tick_kwargs=tick_kwargs)
        if dsw_path_kwargs is None:
            dsw_path_kwargs = {
                'linestyle': '-', 'marker': 'o', 'alpha': 0.5,
                'color': 'darkgreen', 'linewidth': 3
            }
        axs[0].plot(path[:, 1], path[:, 0], **dsw_path_kwargs)

        if showdist:
            axs[3].text(0, 0.2, f"Dist_a = {dist_approx:.4f}")

        if show_values:
            # Print segment statistics
            rdist, rdists = self.distance(per_segment=True)
            adist, adists = self.distance_approx(per_segment=True)
            for segment, srdist, sadist in zip(self.segments, rdists, adists):
                axs[0].text((segment.s_idx_y + segment.e_idx_y)/2 + 4,
                            (segment.s_idx + segment.e_idx) / 2 + 4,
                            f"{srdist:.4f} / {sadist:.4f} - "
                            f"({segment.s_idx},{segment.s_idx_y})",
                            color=dsw_path_kwargs.get("color", "darkgreen"))
            # Show split metric using a circles
            if self.split_strategy == SplitStrategy.DERIV:
                path_val = np.array(self.path)
                inner_dist, _, _ = innerdistance.inner_dist_fns(self.dtw_settings.inner_dist)
                path_ders = self.get_2ndderiv_in_path(inner_dist, self.path)
                path_ders = 200 * (path_ders / np.max(path_ders))
                axs[0].scatter(path_val[:, 1], path_val[:, 0], s=path_ders, c='red', alpha=0.2)
            elif self.split_strategy == SplitStrategy.DERIV_DIST:
                path_val = np.array(self.path)
                inner_dist, _, _ = innerdistance.inner_dist_fns(self.dtw_settings.inner_dist)
                path_ders1 = self.get_1stderiv_in_path(inner_dist, self.path)
                path_ders2 = self.get_2ndderiv_in_path(inner_dist, self.path)
                mean = np.mean(path_val, axis=1)
                dists = np.linalg.norm(path_val - mean[:, np.newaxis], axis=1)
                path_ders = dists*path_ders1 + dists**2*path_ders2
                # path_ders = 200 * (path_ders / np.max(path_ders))
                path_ders12_max = max(np.max(path_ders1), np.max(path_ders2))
                path_ders1 = 200 * (path_ders1 / path_ders12_max)
                path_ders2 = 200 * (path_ders2 / path_ders12_max)
                # axs[0].scatter(path_val[:, 1], path_val[:, 0], s=path_ders, c='red', alpha=0.2)
                axs[0].scatter(path_val[:, 1], path_val[:, 0], s=path_ders2,
                               facecolors='none', edgecolors='yellow', alpha=0.3)
                axs[0].scatter(path_val[:, 1], path_val[:, 0], s=path_ders1,
                               facecolors='none', edgecolors='orange', alpha=0.3)
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_inset = inset_axes(axs[0], width="60%", height="15%", loc='upper right')
                ax_inset.plot(path_ders1, color='blue', alpha=0.5, label='1st')
                ax_inset.plot(path_ders2, color='red', alpha=0.5, label='2nd')
                ax_inset.set_title('1st and 2nd derivative of path', fontsize=8)
                ax_inset.legend(loc='upper right')
                ax_inset.set_xlim((0, len(path_ders1)))

        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
            fig, axs = None, None
        return fig, axs

    def plot_one_segment(self, index_of_intermediate, before_split=True, tick_kwargs=None, filename=None):
        import matplotlib.pyplot as plt
        assert self.save_intermediates and index_of_intermediate < len(self.intermediates)
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
            F = perpendicular_foot([path[mid][1], path[mid][0]], [path[start][1], path[start][0]],
                                   [path[end][1], path[end][0]])  # perpendicular foot
            axs[0].plot([path[mid][1], F[0]], [path[mid][0], F[1]], '-.', 'g', lw=2)  # perpendicular line
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
        axs[0].legend(fontsize=20, loc='lower left')
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


class SortedList:
    def __init__(self, values):
        self._l = sorted(values)

    def __len__(self):
        return self._l.__len__()

    def __iter__(self):
        return self._l.__iter__()

    def __getitem__(self, item):
        return self._l.__getitem__(item)

    def remove(self, x):
        """Remove the value equal to x."""
        try:
            i = self.index(x)
            del self._l[i]
        except ValueError:
            pass

    def index(self, x):
        """Locate the leftmost value exactly equal to x."""
        i = bisect_left(self._l, x)
        if i != len(self._l) and self._l[i] == x:
            return i
        raise ValueError

    def find_lt(self, x):
        """Find rightmost value less than x."""
        i = bisect_left(self._l, x)
        if i:
            return self._l[i - 1]
        raise ValueError

    def find_le(self, x):
        """Find rightmost value less than or equal to x."""
        i = bisect_right(self._l, x)
        if i:
            return self._l[i - 1]
        raise ValueError

    def find_gt(self, x):
        """Find leftmost value greater than x."""
        i = bisect_right(self._l, x)
        if i != len(self._l):
            return self._l[i]
        raise ValueError

    def find_ge(self, x):
        """Find leftmost item greater than or equal to x."""
        i = bisect_left(self._l, x)
        if i != len(self._l):
            return self._l[i]
        raise ValueError

    def __str__(self):
        return str(self._l)

    def __repr__(self):
        return str(self._l)


def plot_warping(
        s1,
        s2,
        segments,
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
):
    """

    :param s1:
    :param s2:
    :param segments:
    :param path:
    :param filename:
    :param fig:
    :param axs:
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
    :return: fig, axs
    """
    import matplotlib as mpl
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Check arguments
    if isinstance(segments, ExplainPair):
        segments = segments.segments
        path = segments.path
    elif path is None:
        raise AttributeError(
            f"Argument path cannot be None if third argument is not an ExplainPair."
        )
    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex="all", sharey="all")
    elif fig is None or axs is None:
        raise TypeError(
            f"The fig and axs arguments need to be both None or both instantiated."
        )

    # Set up axes and plot series
    if series_line_options is None:
        series_line_options = {}
    s1_min = np.min(s1)
    s2_max = np.max(s2)
    s_xlim = max(len(s1) - 1, len(s2) - 1)
    axs[0].plot(s1, **series_line_options)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_position(('outward', 10))
    axs[0].set_xlim(0, s_xlim)
    axs[1].plot(s2, **series_line_options)
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
    plt.tight_layout()

    # Plot amplitude differences
    variations_shade_options = dict(
        color=color_shade,
        alpha=0.4,
        linewidth=0,
    )
    if variations is not None:
        axs[0].fill_between(
            range(len(s1)),
            s1 - variations[:, 0],
            s1 + variations[:, 1],
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
        norm = mpl.colors.Normalize(vmin=0, vmax=np.pi / 2)
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
        cax = fig.add_axes([0.75, 0.90, 0.2, 0.03], zorder=10)
        cax.set_facecolor('white')
        cbar = fig.colorbar(angle_to_rgba_map,
                            ticks=[0.05 * np.pi, 0.25 * np.pi, 0.45 * np.pi],
                            cax=cax, orientation='horizontal', alpha=elasticity_shade_options['alpha'])
        cbar.ax.set_xticklabels([f'{np.tan(0.05 * np.pi):.2f}', '0', f'{np.tan(0.45 * np.pi):.2f}'])
        for tick in cbar.ax.get_xticklabels():
            tick.set_alpha(alpha=elasticity_shade_options['alpha'])
            # tick.set_verticalalignment("top")
            bbox_pixels = tick.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_fig = bbox_pixels.transformed(fig.transFigure.inverted())
            fig.text(bbox_fig.x0, bbox_fig.y0, "x", ha='right', va='bottom',
                     fontsize=tick.get_fontsize() / 1.5, alpha=elasticity_shade_options['alpha'],
                     transform=fig.transFigure)
    else:
        angle_to_rgba = None
        elasticity_shade_options = None

    # Plot segments
    for segment in segments:
        assert isinstance(segment, PathSegment)
        r_c = segment.s_idx
        c_c = path[segment.s_idx_p][1]
        if r_c < 0 or c_c < 0:
            continue
        if start_on_curve:
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1[r_c]],
                coordsA=axs[0].transData,
                xyB=[c_c, s2[c_c]],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
        else:
            # Use vertical lines to the edge of the plot and only then show the warping.
            if continue_lines:
                con = mpatches.ConnectionPatch(
                    xyA=[r_c, s1_max],
                    coordsA=axs[0].transData,
                    xyB=[r_c, s1[r_c]],
                    coordsB=axs[0].transData,
                    **warping_line_options_ext,
                )
                lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1[r_c]],
                coordsA=axs[0].transData,
                xyB=[r_c, s1_min],
                coordsB=axs[0].transData,
                **warping_line_options,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1_min],
                coordsA=axs[0].transData,
                xyB=[c_c, s2_max],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[c_c, s2_max],
                coordsA=axs[1].transData,
                xyB=[c_c, s2[c_c]],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
            if continue_lines:
                con = mpatches.ConnectionPatch(
                    xyA=[c_c, s2[c_c]],
                    coordsA=axs[1].transData,
                    xyB=[c_c, s2_min],
                    coordsB=axs[1].transData,
                    **warping_line_options_ext,
                )
                lines.append(con)
            if show_elasticity and not color_elasticity:
                if segment.expansion > 0:
                    length = segment.length() - 1
                    # coords = [
                    #     axs[0].transData.transform([r_c + length, s1_min]),
                    #     axs[1].transData.transform([c_c + length, s2_max]),
                    #     axs[1].transData.transform([path[segment.e_idx_p][1], s2_max]),
                    #     ]
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
                    # coords = [
                    #     axs[0].transData.transform([segment.e_idx, s1_min]),
                    #     axs[1].transData.transform([path[segment.e_idx_p][1], s2_max]),
                    #     axs[0].transData.transform([segment.e_idx + length_y - length, s1_min]),
                    #     ]
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
                    axs[1].transData.transform([path[segment.e_idx_p][1], s2_max]),
                    axs[1].transData.transform([path[segment.s_idx_p][1], s2_max]),
                ]
                coords = fig.transFigure.inverted().transform(coords)
                elasticity_shade_options['facecolor'] = angle_to_rgba(segment.angle)
                poly = mpatches.Polygon(coords, closed=True, transform=fig.transFigure,
                                        **elasticity_shade_options)
                polys.append(poly)
                compr_ratio = segment.compression_ratio()  # == tan(segment.angle)
                annotation = f"{compr_ratio:.2f}"
                text_obj = fig.text((coords[2][0] + coords[3][0]) / 2,
                                    (coords[0][1] + coords[3][1]) / 2 - elasticity_line_height,
                                    annotation, ha='center', va='baseline',
                                    alpha=elasticity_shade_options['alpha'])
                bbox_text = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
                bbox_poly = poly.get_window_extent(renderer=fig.canvas.get_renderer())

                if bbox_text.x0 < bbox_poly.x0 or bbox_text.x1 > bbox_poly.x1:
                    text_obj.remove()
                else:
                    bbox_fig = fig.transFigure.inverted().transform([(bbox_text.x0, bbox_text.y0)])
                    fig.text(bbox_fig[0][0], (coords[0][1] + coords[3][1]) / 2 - elasticity_line_height,
                             "x", ha='right', va='baseline',
                             fontsize=text_obj.get_fontsize() / 1.5,
                             alpha=elasticity_shade_options['alpha'])

        r_c = segment.e_idx
        c_c = path[segment.e_idx_p][1]
        if r_c < 0 or c_c < 0:
            continue
        # Also draw last line (at end of segment instead of begin)
        if start_on_curve:
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1[r_c]],
                coordsA=axs[0].transData,
                xyB=[c_c, s2[c_c]],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
        else:
            if continue_lines:
                con = mpatches.ConnectionPatch(
                    xyA=[r_c, s1_max],
                    coordsA=axs[0].transData,
                    xyB=[r_c, s1[r_c]],
                    coordsB=axs[0].transData,
                    **warping_line_options_ext,
                )
                lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1[r_c]],
                coordsA=axs[0].transData,
                xyB=[r_c, s1_min],
                coordsB=axs[0].transData,
                **warping_line_options,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1_min],
                coordsA=axs[0].transData,
                xyB=[c_c, s2_max],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[c_c, s2_max],
                coordsA=axs[1].transData,
                xyB=[c_c, s2[c_c]],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
            if continue_lines:
                con = mpatches.ConnectionPatch(
                    xyA=[c_c, s2[c_c]],
                    coordsA=axs[1].transData,
                    xyB=[c_c, s2_min],
                    coordsB=axs[1].transData,
                    **warping_line_options_ext,
                )
                lines.append(con)
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


def plot_warping2(
        s1,
        s2,
        segments,
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
):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if isinstance(segments, ExplainPair):
        segments = segments.segments
        path = segments.path
    elif path is None:
        raise AttributeError(
            f"Argument path cannot be None if third argument is not an ExplainPair."
        )
    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex="all", sharey="all")
    elif fig is None or axs is None:
        raise TypeError(
            f"The fig and axs arguments need to be both None or both instantiated."
        )

    # Plot series
    if series_line_options is None:
        series_line_options = {'color': color_series}
    s1_min = np.min(s1)
    s2_max = np.max(s2)
    axs[0].plot(s1, **series_line_options)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_position(('outward', 10))
    axs[0].set_xlim(0)
    axs[1].plot(s2, **series_line_options)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_position(('outward', 10))
    axs[1].set_xlim(0)
    if not show_xticks:
        axs[0].set_xticks([])
    if not show_yticks:
        axs[0].set_yticks([])
        axs[1].set_yticks([])
    plt.tight_layout()

    # Amplitude differences
    variations_shade_options = dict(
        color=color_shade,
        alpha=0.4,
        linewidth=0,
    )
    if variations is not None:
        axs[0].fill_between(
            range(len(s1)),
            s1 - variations[:, 0],
            s1 + variations[:, 1],
            **variations_shade_options
        )
        axs[0].plot(range(len(s1)), s1 - variations[:, 0], color=color_shade)
        axs[0].plot(range(len(s1)), s1 + variations[:, 1], color=color_shade)
        axs[0].plot(s1, **series_line_options)

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

    for segment in segments:
        assert isinstance(segment, PathSegment)
        r_c = segment.s_idx
        c_c = path[segment.s_idx_p][1]
        if r_c < 0 or c_c < 0:
            continue
        if start_on_curve:
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1[r_c]],
                coordsA=axs[0].transData,
                xyB=[c_c, s2[c_c]],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
        else:
            # Use vertical lines to the edge of the plot and only then show the warping.
            if continue_lines:
                con = mpatches.ConnectionPatch(
                    xyA=[r_c, s1_max],
                    coordsA=axs[0].transData,
                    xyB=[r_c, s1[r_c]],
                    coordsB=axs[0].transData,
                    **warping_line_options_ext,
                )
                lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1[r_c]],
                coordsA=axs[0].transData,
                xyB=[r_c, s1_min],
                coordsB=axs[0].transData,
                **warping_line_options,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1_min],
                coordsA=axs[0].transData,
                xyB=[c_c, s2_max],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[c_c, s2_max],
                coordsA=axs[1].transData,
                xyB=[c_c, s2[c_c]],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
            if continue_lines:
                con = mpatches.ConnectionPatch(
                    xyA=[c_c, s2[c_c]],
                    coordsA=axs[1].transData,
                    xyB=[c_c, s2_min],
                    coordsB=axs[1].transData,
                    **warping_line_options_ext,
                )
                lines.append(con)
            if show_elasticity:
                compr_ratio = segment.compression_logratio()
                annotation = f"{np.exp(compr_ratio):.2f}x"
                if segment.expansion > 0:
                    arrowstyle = "<->"
                elif segment.compression > 0:
                    arrowstyle = ">-<"
                else:
                    arrowstyle = "---"
                coords = [
                    axs[1].transData.transform([path[segment.s_idx_p][1] + 1, s2_max]),
                    axs[1].transData.transform([path[segment.e_idx_p][1] - 1, s2_max]),
                ]
                coords = fig.transFigure.inverted().transform(coords)
                coords[0, 1] -= 1.2 * elasticity_line_height
                coords[1, 1] -= 1.2 * elasticity_line_height
                fig.text((coords[0, 0] + coords[1, 0]) / 2, coords[0, 1] + 1.2 * elasticity_line_height,
                         annotation,
                         horizontalalignment="center", color=color_shade, alpha=0.6,
                         transform=fig.transFigure)
                x, y = zip(*coords)
                poly = _plot_arrow2(x, y, arrowstyle, elasticity_line_height, fig)
                polys.append(poly)

        r_c = segment.e_idx
        c_c = path[segment.e_idx_p][1]
        if r_c < 0 or c_c < 0:
            continue
        # Also draw last line (at end of segment instead of begin)
        if start_on_curve:
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1[r_c]],
                coordsA=axs[0].transData,
                xyB=[c_c, s2[c_c]],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
        else:
            if continue_lines:
                con = mpatches.ConnectionPatch(
                    xyA=[r_c, s1_max],
                    coordsA=axs[0].transData,
                    xyB=[r_c, s1[r_c]],
                    coordsB=axs[0].transData,
                    **warping_line_options_ext,
                )
                lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1[r_c]],
                coordsA=axs[0].transData,
                xyB=[r_c, s1_min],
                coordsB=axs[0].transData,
                **warping_line_options,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[r_c, s1_min],
                coordsA=axs[0].transData,
                xyB=[c_c, s2_max],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
            con = mpatches.ConnectionPatch(
                xyA=[c_c, s2_max],
                coordsA=axs[1].transData,
                xyB=[c_c, s2[c_c]],
                coordsB=axs[1].transData,
                **warping_line_options,
            )
            lines.append(con)
            if continue_lines:
                con = mpatches.ConnectionPatch(
                    xyA=[c_c, s2[c_c]],
                    coordsA=axs[1].transData,
                    xyB=[c_c, s2_min],
                    coordsB=axs[1].transData,
                    **warping_line_options_ext,
                )
                lines.append(con)
    for poly in polys:
        fig.patches.append(poly)
    for line in lines:
        fig.add_artist(line)
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, axs = None, None
    return fig, axs


def plot_explain(series_from, segments, variations=None, filename=None, fig=None):
    import matplotlib.pyplot as plt

    process = list(enumerate(segments))
    cur_row = 0
    # shift, expansion, compression
    plot_row = np.zeros((len(segments), 3))
    while len(process) != 0:
        s_i, e_i, c_i = 0, 0, 0
        postpone = []
        for segi, segment in process:
            if isinstance(segment, PathSegment):
                segment = segment.segment
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
    ax.set_title("Shift + Elasticity")
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
        if isinstance(segments[idx], PathSegment):
            segment = segments[idx].segment
        else:
            segment = segments[idx]
        current_max = max(segment.shift_l, segment.shift_r, round(segment.length() * np.tan(segment.a_compression) / 2),
                          round(segment.length() * np.tan(segment.a_expansion) / 2))
        current_min = min(segment.shift_l, segment.shift_r, round(segment.length() * np.tan(segment.a_compression) / 2),
                          round(segment.length() * np.tan(segment.a_expansion) / 2))
        max_shift_or_expansion = max(max_shift_or_expansion, current_max)
        min_shift_or_expansion = min(min_shift_or_expansion, current_min)

    for idx in range(len(segments)):
        if isinstance(segments[idx], PathSegment):
            segment = segments[idx].segment
        else:
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
        # y1 = [r + 0.25, r + 0.25]
        # y2 = [r, r]
        # axs[1].fill_between(x, y1, y2, color=color_shade, alpha=0.5, linewidth=0)
        y = [r - 0.15, r - 0.15]
        _plot_arrow(x, y, "<-<", segment.shift_l, min_shift_or_expansion, max_shift_or_expansion, ax)
        x = [bi + segment.shift_r, ei + segment.shift_r]
        # axs[1].fill_between(x, y1, y2, color=color_shade, alpha=0.5, linewidth=0)
        y = [r - 0.3, r - 0.3]
        _plot_arrow(x, y, ">->", segment.shift_r, min_shift_or_expansion, max_shift_or_expansion, ax)

        # Compression
        delta = round(segment.length() * np.tan(segment.a_compression) / 2)
        # x = [bi, bi + delta, ei - delta, ei]
        # y1 = [r + 0.25, r + 0.45, r + 0.45, r + 0.25]
        # y2 = [r + 0.25, r + 0.25, r + 0.25, r + 0.25]
        # axs[1].fill_between(x, y1, y2, color=color_shade, alpha=0.5, linewidth=0)
        x = [bi + delta, ei - delta]
        y = [r + 0.4, r + 0.4]
        _plot_arrow(x, y, ">-<", delta, min_shift_or_expansion, max_shift_or_expansion, ax)

        # Expansion
        delta = round(segment.length() * np.tan(segment.a_expansion) / 2)
        # x = [bi - delta, bi, ei, ei + delta]
        # y1 = [r - 0.20, r, r, r - 0.20]
        # y2 = [r - 0.20, r - 0.20, r - 0.20, r - 0.20]
        # axs[1].fill_between(x, y1, y2, color=color_shade, alpha=0.5, linewidth=0)
        x = [bi - delta, ei + delta]
        y = [r + 0.55, r + 0.55]
        _plot_arrow(x, y, "<->", delta, min_shift_or_expansion, max_shift_or_expansion, ax)

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


def _plot_arrow(x, y, arrowstyle, delta, delta_min, delta_max, ax):
    """Custom arrow plotting. Allows for inwards pointing arrows.

    :param x: A pair of x coordinates
    :param y: A pair of y coordinates
    :param arrowstyle: One of '<->', '<-<', '>->', '>-<'
    :param delta: Amount of change. Used to determine the color transparency.
    :param delta_min: Minimum amount of change. Used to determine the color transparency.
    :param delta_max: Maximum amount of change. Used to determine the color transparency.
    :param ax: Matplotlib Axes
    :return:
    """
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath

    t = ax.transAxes.transform([(0, 0), (1, 1)])
    t = ax.get_figure().get_dpi() / (t[1, 1] - t[0, 1]) / 23
    mdx, mdy = 144 * t, 0.10
    ml, _, mr = arrowstyle

    if ml == ">":
        sl = [
            (x[0], y[0] + mdy),
            (x[0] + mdx, y[0]),
            (x[0], y[0] - mdy),
            (x[0] + mdx, y[0]),
        ]
    elif ml == "<":
        sl = [
            (x[0] + mdx, y[0] - mdy),
            (x[0], y[0]),
            (x[0] + mdx, y[0] + mdy),
            (x[0], y[0]),
        ]
    else:
        raise ValueError(f'Unknown ml: {ml}')
    if mr == ">":
        sr = [
            (x[1], y[1]),
            (x[1] - mdx, y[1] + mdy),
            (x[1] - mdx, y[1] - mdy),
            (x[1], y[1]),
        ]
    elif mr == "<":
        sr = [
            (x[1] - mdx, y[1]),
            (x[1], y[1] + mdy),
            (x[1] - mdx, y[1]),
            (x[1], y[1] - mdy),
        ]
    else:
        raise ValueError(f'Unknown mr: {mr}')

    alpha = 0.5 * (delta - delta_min) / (delta_max - delta_min) + 0.2
    # ax.plot(x, y, color=color_shade, alpha=alpha)
    # ax.plot(x[0], y[0], color=color_shade, marker=ml, alpha=alpha, markersize=marker_size)
    # ax.plot(x[1], y[1], color=color_shade, marker=mr, alpha=alpha, markersize=marker_size)
    pp1 = mpatches.PathPatch(
        mpath.Path(
            sl + sr,
            [
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
            ],
        ),
        fc="none",
        transform=ax.transData,
        color=color_shade,
        alpha=alpha,
        linewidth=1,
        clip_on=False,
    )
    ax.add_patch(pp1)


def _plot_arrow2(x, y, arrowstyle, height, fig):
    """Custom arrow plotting. Allows for inwards pointing arrows.

    :param x: A pair of x coordinates
    :param y: A pair of y coordinates
    :param arrowstyle: One of '<->', '<-<', '>->', '>-<', '---'
    :param delta: Amount of change. Used to determine the color transparency.
    :param epsilon: Amount of ignored change. Used to determine the color transparency.
    :param ax: Matplotlib Axes
    :return:
    """
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath

    # t = ax.transAxes.transform([(0, 0), (1, 1)])
    # t = ax.get_figure().get_dpi() / (t[1, 1] - t[0, 1]) / 23
    mdx, mdy = height, height
    ml, _, mr = arrowstyle

    if ml == ">":
        sl = [
            (x[0], y[0] + mdy),
            (x[0] + mdx, y[0]),
            (x[0], y[0] - mdy),
            (x[0] + mdx, y[0]),
        ]
    elif ml == "<":
        sl = [
            (x[0] + mdx, y[0] - mdy),
            (x[0], y[0]),
            (x[0] + mdx, y[0] + mdy),
            (x[0], y[0]),
        ]
    elif ml == "-":
        sl = [
            (x[0], y[0]),
            (x[0], y[0]),
            (x[0], y[0]),
            (x[0], y[0]),
        ]
    else:
        raise ValueError(f'Unknown ml: {ml}')
    if mr == ">":
        sr = [
            (x[1], y[1]),
            (x[1] - mdx, y[1] + mdy),
            (x[1] - mdx, y[1] - mdy),
            (x[1], y[1]),
        ]
    elif mr == "<":
        sr = [
            (x[1] - mdx, y[1]),
            (x[1], y[1] + mdy),
            (x[1] - mdx, y[1]),
            (x[1], y[1] - mdy),
        ]
    elif mr == "-":
        sr = [
            (x[1], y[1]),
            (x[1], y[1]),
            (x[1], y[1]),
            (x[1], y[1]),
        ]
    else:
        raise ValueError(f'Unknown mr: {mr}')

    alpha = 0.6
    # ax.plot(x, y, color=color_shade, alpha=alpha)
    # ax.plot(x[0], y[0], color=color_shade, marker=ml, alpha=alpha, markersize=marker_size)
    # ax.plot(x[1], y[1], color=color_shade, marker=mr, alpha=alpha, markersize=marker_size)
    pp1 = mpatches.PathPatch(
        mpath.Path(
            sl + sr,
            [
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
            ],
        ),
        fc="none",
        transform=fig.transFigure,
        color=color_shade,
        alpha=alpha,
        linewidth=1,
        clip_on=False,
    )
    return pp1


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


def rdp_vectorized(points, epsilon):
    """
    Ramer-Douglas-Peucker algorithm to simplify a path.

    :param points: 2D numpy array (points x dimensions)
    :param epsilon: Maximum deviation between the original curve and the simplified curve
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    queue = deque([(0, len(points) - 1)])
    result = set()

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

        idxmax = np.argmax(distances)
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
    result = sorted(result)
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
