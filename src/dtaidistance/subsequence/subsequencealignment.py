# -*- coding: UTF-8 -*-
"""
dtaidistance.subsequence.subsequencealignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
from .. import innerdistance



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


def subsequence_alignment(query, series, penalty=0.1, use_c=False):
    """See SubsequenceAligment.

    :param query:
    :param series:
    :param penalty:
    :param use_c:
    :return:
    """
    sa = SubsequenceAlignment(query, series, penalty=penalty, use_c=use_c)
    sa.align()
    return sa


class SAMatch:
    def __init__(self, idx, alignment):
        """SubsequenceAlignment match"""
        self.idx = idx
        self.alignment = alignment
        self._path = None
        self._segment = None

    @property
    def value(self):
        """Normalized DTW distance of match.

        Normalization is the DTW distance divided by the query length.
        """
        return self.alignment.matching[self.idx]

    @property
    def distance(self):
        """DTW distance of match.

        This value is dependent on the length of the query. Use the value
        property when comparing queries of different lengths.
        """
        return self.value * len(self.alignment.query)

    @property
    def segment(self):
        """Matched segment in series."""
        if self._segment is None:
            start = self.alignment.matching_function_startpoint(self.idx, path=self.path)
            end = self.alignment.matching_function_endpoint(self.idx)
            self._segment = [start, end]
        return self._segment

    @property
    def path(self):
        """Matched path in series"""
        if self._path is None:
            self._path = self.alignment.matching_function_bestpath(self.idx)
        return self._path

    def __str__(self):
        return f'SAMatch({self.idx})'

    def __repr__(self):
        return self.__str__()


class SubsequenceAlignment:
    def __init__(self, query, series, penalty=0.1, use_c=False, **kwargs):
        """Subsequence alignment using DTW.
        Find where the query occurs in the series.

        Based on Fundamentals of Music Processing, Meinard Müller, Springer, 2015.

        Example::

            query = np.array([1., 2, 0])
            series = np.array([1., 0, 1, 2, 1, 0, 2, 0, 3, 0, 0])
            sa = subsequence_search(query, series)
            mf = sa.matching_function()
            sa.kbest_matches(k=2)


        :param query: Subsequence to search for
        :param series: Long sequence in which to search
        :param penalty: Penalty for non-diagonal matching
        :param use_c: Use the C-based DTW function if available
        """
        self.query = query
        self.series = series
        self.paths = None
        self.matching = None
        self.use_c = use_c
        self.settings = dtw.DTWSettings(penalty=penalty,
                                        psi=[0, 0, len(self.series), len(self.series)],
                                        **kwargs)

    def reset(self):
        self.matching = None

    def align(self):
        if self.matching is not None:
            return
        if np is not None and isinstance(self.series, np.ndarray) and len(self.series.shape) > 1:
            if not self.use_c:
                _, self.paths = dtw_ndim.warping_paths(self.query, self.series,
                                                       penalty=self.settings.penalty, psi=self.settings.psi,
                                                       psi_neg=False, keep_int_repr=True)
            else:
                _, self.paths = dtw_ndim.warping_paths_fast(self.query, self.series,
                                                            penalty=self.settings.penalty, psi=self.settings.psi,
                                                            compact=False, psi_neg=False, keep_int_repr=True)
        else:
            if not self.use_c:
                _, self.paths = dtw.warping_paths(self.query, self.series,
                                                  penalty=self.settings.penalty, psi=self.settings.psi,
                                                  psi_neg=False, keep_int_repr=True)
            else:
                _, self.paths = dtw.warping_paths_fast(self.query, self.series,
                                                       penalty=self.settings.penalty, psi=self.settings.psi,
                                                       compact=False, psi_neg=False, keep_int_repr=True)
        self._compute_matching()

    def align_fast(self):
        use_c = self.use_c
        self.use_c = True
        result = self.align()
        self.use_c = use_c
        return result

    def _compute_matching(self):
        idist_fn, result_fn, _ = innerdistance.inner_dist_fns(self.settings.inner_dist,
                                                              use_ndim=self.settings.use_ndim)
        matching = self.paths[-1, :]
        if len(matching) > len(self.series):
            matching = result_fn(matching[-len(self.series):])
        self.matching = np.array(matching) / len(self.query)

    def warping_paths(self):
        """Get matrix with all warping paths.

        If the aligmnent was computed using a compact, the paths are first copied into a full
        warping paths matrix.

        :return: Numpy matrix of size (len(query)+1) * (len(series)+1)
        """
        return self.paths

    def matching_function(self):
        """The matching score for each end-point of a possible match."""
        return self.matching

    def get_match(self, idx):
        return SAMatch(idx, self)

    def best_match_fast(self):
        use_c = self.use_c
        self.use_c = True
        result = self.best_match()
        self.use_c = use_c
        return result

    def best_match(self):
        best_idx = np.argmin(self.matching)
        return self.get_match(best_idx)

    def kbest_matches_fast(self, *args, **kwargs):
        """See :meth:`kbest_matches`."""
        use_c = self.use_c
        self.use_c = True
        result = self.kbest_matches(*args, **kwargs)
        self.use_c = use_c
        return result

    def kbest_matches(self, k=1, overlap=0, minlength=2, maxlength=None):
        """Yields the next best match. Stops at k matches (use None for all matches).

        :param k: Number of matches to yield. None is all matches.
        :param overlap: Matches cannot overlap unless overlap > 0.
        :param minlength: Minimal length of the matched sequence.
            If k is set to None, matches with one value can occur if minlength is set to 1.
        :param maxlength: Maximal length of the matched sequence.
        :return: Yield an SAMatch object
        """
        return self._best_matches(k=k, overlap=overlap,
                                  minlength=minlength, maxlength=maxlength)

    def best_matches_fast(self, *args, **kwargs):
        """See :meth:`best_matches`."""
        use_c = self.use_c
        self.use_c = True
        result = self.best_matches(*args, **kwargs)
        self.use_c = use_c
        return result

    def best_matches(self, max_rangefactor=2, overlap=0, minlength=2, maxlength=None):
        """Yields the next best match. Stops when the current match is larger than
        maxrangefactor times the first match.

        :param max_rangefactor: The range between the first (best) match and the last match
            can be at most a factor of ``maxrangefactor``. For example, if the first match has
            value v_f, then the last match has a value ``v_l < v_f*maxfactorrange``.
        :param overlap: Matches cannot overlap unless ``overlap > 0``.
        :param minlength: Minimal length of the matched sequence.
        :param maxlength: Maximal length of the matched sequence.
        :return:
        """
        return self._best_matches(k=None, max_rangefactor=max_rangefactor,
                                  detectknee_alpha=None,
                                  overlap=overlap,
                                  minlength=minlength, maxlength=maxlength)

    def best_matches_knee_fast(self, *args, **kwargs):
        """See :meth:`best_matches_knee`."""
        use_c = self.use_c
        self.use_c = True
        result = self.best_matches_knee(*args, **kwargs)
        self.use_c = use_c
        return result

    def best_matches_knee(self, alpha=0.3, overlap=0, minlength=2, maxlength=None):
        """Yields the next best match. Stops when the current match is larger than
        maxrangefactor times the first match.

        :param alpha: The factor for the exponentially moving average that keeps
            track of the curve to detect the knee. The higher, the more sensitive
            to recent values (and differences).
        :param overlap: Matches cannot overlap unless ``overlap > 0``.
        :param minlength: Minimal length of the matched sequence.
        :param maxlength: Maximal length of the matched sequence.
        :return:
        """
        return self._best_matches(k=None, max_rangefactor=None,
                                  detectknee_alpha=alpha,
                                  overlap=overlap,
                                  minlength=minlength, maxlength=maxlength)

    def _best_matches(self, k=None, overlap=0, minlength=2, maxlength=None,
                      max_rangefactor=None, detectknee_alpha=None):
        self.align()
        matching = np.array(self.matching)
        maxv = np.ceil(np.max(matching) + 1)
        matching[:min(len(self.query) - 1, overlap)] = maxv
        ki = 0
        max_dist = np.inf
        if detectknee_alpha is not None:
            dk = util.DetectKnee(alpha=detectknee_alpha)
        else:
            dk = None
        while k is None or ki < k:
            best_idx = np.argmin(matching)
            if np.isinf(matching[best_idx]) or matching[best_idx] == maxv:
                # No more matches found
                break
            if max_rangefactor is not None:
                if ki == 0:
                    max_dist = matching[best_idx] * max_rangefactor
                elif matching[best_idx] > max_dist:
                    # Remaining matches are larger than a factor of the first match
                    break
            if detectknee_alpha is not None:
                if dk.dostop(matching[best_idx]):
                    break
            match = self.get_match(best_idx)
            b, e = match.segment
            cur_overlap = min(overlap, e - b - 1)
            mb, me = best_idx + 1 - (e - b) + cur_overlap, best_idx + 1
            if ((minlength is not None and e-b+1 < minlength) or
                    (maxlength is not None and e-b+1 > maxlength)):
                # Found sequence is too short or too long
                matching[best_idx] = maxv
                continue
            if np.isinf(np.max(matching[mb:me])):
                # No overlapping matches
                matching[best_idx] = maxv
                continue
            matching[mb:me] = np.inf
            ki += 1
            yield match

    def matching_function_segment(self, idx):
        """Matched segment in series."""
        start = self.matching_function_startpoint(idx)
        end = self.matching_function_endpoint(idx)
        return [start, end]

    def matching_function_endpoint(self, idx):
        """Index in series for end of match in matching function at idx.

        :param idx: Index in matching function
        :return: Index in series
        """
        if len(self.matching) == len(self.series):
            return idx
        diff = len(self.series) - len(self.matching)
        return idx + diff

    def matching_function_startpoint(self, idx, path=None):
        """Index in series for start of match in matching function at idx.

        :param idx: Index in matching function
        :return: Index in series
        """
        if path is None:
            path = self.matching_function_bestpath(idx)
        start_idx = path[0][1]
        return start_idx

    def matching_function_bestpath(self, idx):
        """Indices in series for best path for match in matching function at idx.

        :param idx: Index in matching function
        :return: List of (row, col)
        """
        real_idx = idx + 1
        path = dtw.best_path(self.paths, col=real_idx, penalty=self.settings.adj_penalty)
        return path
