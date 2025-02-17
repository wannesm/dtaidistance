# -*- coding: UTF-8 -*-
"""
dtaidistance.clustering.hierarchical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Time series clustering using hierarchical clustering.

:author: Wannes Meert
:copyright: Copyright 2017-2022 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import math
import logging
from pathlib import Path
from collections import deque

try:
    import numpy as np
except ImportError:
    np = None

from dtaidistance.util import SeriesContainer
from dtaidistance.exceptions import NumpyException, MatplotlibException, ScipyException

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger("be.kuleuven.dtai.distance")


class Hierarchical:
    """Hierarchical clustering.

    Note: This method first computes the entire distance matrix. This is not ideal for extremely large
    data sets.

    :param dists_fun: Function to compute pairwise distance matrix between set of series.
    :param dists_options: Arguments to pass to dists_fun.
    :param max_dist: Do not merge or cluster series that are further apart than this.
    :param merge_hook: Function that is called when two series are clustered.
        The function definition is `def merge_hook(from_idx, to_idx, distance)`, where idx is the index of the series.
    :param order_hook: Function that is called to decide on the next idx out of all shortest distances
    :param show_progress: Use a tqdm progress bar
    :return: Cluster indices
    """

    def __init__(self, dists_fun, dists_options, max_dist=float('inf'),
                 merge_hook=None, order_hook=None, show_progress=True):
        self.dists_fun = dists_fun
        self.dists_options = dists_options
        self.max_dist = max_dist
        self.merge_hook = merge_hook
        self.order_hook = order_hook
        self.show_progress = show_progress

    def fit(self, series):
        """Merge sequences.

        :param series: Sequence over series.
        :return: Dictionary with as keys the prototype indicices and as values all the indicides of the series in
            that cluster.
        """
        if np is None:
            raise NumpyException("The fit function requires Numpy to be installed.")
        nb_series = len(series)
        cluster_idx = dict()
        self.dists_options['only_triu'] = True
        dists = self.dists_fun(series, **self.dists_options)
        min_value = np.min(dists)
        min_idxs = np.argwhere(dists == min_value)
        if self.order_hook:
            min_idx = self.order_hook(min_idxs)
        else:
            min_idx = min_idxs[0, :]
        deleted = set()
        cnt_merge = 0
        logger.debug('Merging patterns')
        if self.show_progress and tqdm:
            pbar = tqdm(total=dists.shape[0])
        else:
            pbar = None
        # Hierarchical clustering (distance to prototype)
        while min_value <= self.max_dist and not np.isinf(min_value):
            cnt_merge += 1
            i1, i2 = int(min_idx[0]), int(min_idx[1])
            if self.merge_hook:
                result = self.merge_hook(i2, i1, min_value)
                if result:
                    i1, i2 = result
            logger.debug("Merge {} <- {} ({:.3f})".format(i1, i2, min_value))
            if i1 not in cluster_idx:
                cluster_idx[i1] = {i1}
            if i2 in cluster_idx:
                cluster_idx[i1].update(cluster_idx[i2])
                del cluster_idx[i2]
            else:
                cluster_idx[i1].add(i2)
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
            if len(deleted) == nb_series - 1:
                break
            if pbar:
                pbar.update(1)
            # min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            # min_value = dists[min_idx]
            min_value = np.min(dists)
            # if np.isinf(min_value):
            #     break
            min_idxs = np.argwhere(dists == min_value)
            if self.order_hook:
                min_idx = self.order_hook(min_idxs)
            else:
                min_idx = min_idxs[0, :]
        if pbar:
            pbar.update(dists.shape[0] - cnt_merge)

        prototypes = []
        for i in range(len(series)):
            if i not in deleted:
                prototypes.append(i)
                if i not in cluster_idx:
                    cluster_idx[i] = {i}
        return cluster_idx

    def plot(self, *args, **kwargs):
        raise Exception("Class Hierarchical does not support plotting. "
                        "Use the class HierarchicalTree.")


class BaseTree:
    """Base Tree abstract class.

    Returns a datastructure compatible with the Scipy clustering methods:

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are
    combined to form cluster n + i. A cluster with an index less than n corresponds to one of the original
    observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value
    Z[i, 3] represents the number of original observations in the newly formed cluster.
    """

    def __init__(self, **kwargs):
        self.linkage = None
        self.series = None
        self._series_y = None
        self.ts_height_factor = None

    @property
    def maxnode(self):
        return len(self.series) - 1 + len(self.linkage)

    def get_linkage(self, node):
        if node < len(self.series):
            return None
        idx = int(node - len(self.series))
        return self.linkage[idx]

    def plot(self, filename=None, axes=None, ts_height=10,
             bottom_margin=2, top_margin=2, ts_left_margin=0, ts_sample_length=1,
             tr_label_margin=3, tr_left_margin=2, ts_label_margin=0,
             show_ts_label=None, show_tr_label=None,
             cmap='viridis_r', ts_color=None):
        """Plot the hierarchy and time series.

        :param filename: If a filename is passed, the image is written to this file.
        :param axes: If a axes array is passed the image is added to this figure.
            Expects axes[0] and axes[1] to be present.
        :param ts_height: Height of a time series
        :param bottom_margin: Margin on bottom
        :param top_margin: Margin on top
        :param ts_left_margin: Margin on left of time series image
        :param ts_sample_length: Space between two points in the time series
        :param tr_label_margin: Margin between tree split and label
        :param tr_left_margin: Left margin for tree
        :param ts_label_margin: Margin between start of series and label
        :param show_ts_label: Show label indices. Boolean, callable or subscriptable object.
            If it is a callable object, the index of the time series will be given and the
            return string will be printed.
        :param show_tr_label: Show tree distances. Boolean, callable or subscriptable object.
            If it is a callable object, the index of the time series will be given and the
            return string will be printed.
        :param cmap: Matplotlib colormap name
        :param ts_color: function that takes the index and returns a color
            (compatible with the matplotlib.color color argument)
        """
        # print('linkage')
        # for l in self.linkage:
        #     print(l)
        if np is None:
            raise NumpyException("The plot function requires Numpy to be installed.")
        try:
            from matplotlib import pyplot as plt
            from matplotlib.lines import Line2D
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
        except ImportError:
            raise MatplotlibException("The plot function requires Matplotlib to be installed.")

        if show_ts_label is True:
            show_ts_label = lambda idx: str(int(idx))
        elif show_ts_label is False or show_ts_label is None:
            show_ts_label = lambda idx: ""
        elif callable(show_ts_label):
            pass
        elif hasattr(show_ts_label, "__getitem__"):
            show_ts_label_prev = show_ts_label
            show_ts_label = lambda idx: show_ts_label_prev[idx]
        else:
            raise AttributeError("Unknown type for show_ts_label, expecting boolean, subscriptable or callable, "
                                 "got {}".format(type(show_ts_label)))
        if show_tr_label is True:
            show_tr_label = lambda dist: "{:.2f}".format(dist)
        elif show_tr_label is False or show_tr_label is None:
            show_tr_label = lambda dist: ""
        elif callable(show_tr_label):
            pass
        elif hasattr(show_tr_label, "__getitem__"):
            show_tr_label_prev = show_tr_label
            show_tr_label = lambda idx: show_tr_label_prev[idx]
        else:
            raise AttributeError("Unknown type for show_ts_label, expecting boolean, subscriptable or callable, "
                                 "got {}".format(type(show_ts_label)))

        self._series_y = [0] * len(self.series)

        max_dist = 0
        for _, _, d, _ in self.linkage:
            if not np.isinf(d):
                max_dist = max(max_dist, d)

        node_props = dict()

        min_y, max_y = self.series.get_max_min_y()
        self.ts_height_factor = (ts_height / 2 / max(abs(max_y), abs(min_y)))

        def count(node, height):
            # print('count({},{})'.format(node, height))
            maxheight = None
            maxcumdist = None
            curdepth = None
            cnt = 0
            left_cnt = None
            right_cnt = None
            if node < len(self.series):
                # Leaf
                cnt += 1
                maxheight = height
                maxcumdist = 0
                curdepth = 0
                left_cnt = 0
                right_cnt = 0
            else:
                # Inner node
                child_left, child_right, dist, cnt2 = self.get_linkage(int(node))
                child_left, child_right, cnt2 = int(child_left), int(child_right), int(cnt2)
                if child_left == child_right:
                    raise Exception("Row in linkage contains same node as left and right child: {}-{}".
                                    format(child_left, child_right))
                if np.isinf(dist):
                    dist = 1.5*max_dist
                # Left
                nc, nmh, ncd, nmd = count(child_left, height + 1)
                cnt += nc
                maxheight = nmh
                maxcumdist = nmd + dist
                curdepth = ncd + 1
                left_cnt = nc
                # Right
                nc, nmh, ncd, nmd = count(child_right, height + 1)
                cnt += nc
                maxheight = max(maxheight, nmh)
                maxcumdist = max(maxcumdist, nmd + dist)
                curdepth = max(curdepth, ncd + 1)
                right_cnt = nc
                # if cnt != cnt2:
                #     raise Exception("Count in linkage not correct")
            # print('c', node, c)
            node_props[int(node)] = (cnt, curdepth, left_cnt, right_cnt, maxcumdist)
            # print('count({},{}) = {}, {}, {}, {}'.format(node, height, cnt, maxheight, curdepth, maxcumdist))
            return cnt, maxheight, curdepth, maxcumdist

        cnt, maxheight, curdepth, maxcumdist = count(self.maxnode, 0)
        # for node, props in node_props.items():
        #     print("{:<3}: {}".format(node, props))

        if axes is None:
            fig, ax = plt.subplots(nrows=1, ncols=2, frameon=False)
        else:
            fig, ax = None, axes
        ax[0].set_axis_off()
        # ax[0].set_xlim(left=0, right=curdept)
        ax[0].set_xlim(left=0, right=tr_left_margin + maxcumdist + 0.05)
        ax[0].set_ylim(bottom=0, top=bottom_margin + ts_height * len(self.series) + top_margin)
        # ax[0].plot([0,1],[1,2])
        # ax[0].add_line(Line2D((0,1),(2,2), lw=2, color='black', axes=ax[0]))

        ax[1].set_axis_off()
        max_length = max(len(s) for s in self.series)
        ax[1].set_xlim(left=0, right=ts_left_margin + ts_sample_length * max_length)
        ax[1].set_ylim(bottom=0, top=bottom_margin + ts_height * len(self.series) + top_margin)

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        else:
            pass
        line_colors = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=max_dist), cmap=cmap)

        cnt_ts = 0

        def plot_i(node, depth, cnt_ts, prev_lcnt, ax, left):
            # print('plot_i', node, depth, cnt_ts, prev_lcnt)
            pcnt, pdepth, plcnt, prcnt, pcdist = node_props[node]
            # px = maxheight - pdepth
            px = tr_left_margin + maxcumdist - pcdist
            py = prev_lcnt * ts_height
            if node < len(self.series):
                # Plot series
                # print('plot series y={}'.format(ts_bottom_margin + ts_height * cnt_ts + self.ts_height_factor))
                self._series_y[int(node)] = bottom_margin + ts_height * cnt_ts
                serie = self.series[int(node)]
                ax[1].text(ts_left_margin + ts_label_margin,
                           bottom_margin + ts_height * cnt_ts + ts_height / 2,
                           show_ts_label(int(node)), ha='left', va='center')
                if ts_color:
                    curcolor = ts_color(int(node))
                else:
                    curcolor = None
                line, = ax[1].plot(ts_left_margin + ts_sample_length * np.arange(len(serie)),
                                   bottom_margin + ts_height * (cnt_ts + 0.5) + self.ts_height_factor * serie,
                                   color=curcolor)
                cnt_ts += 1

            else:
                child_left, child_right, dist, _ = self.get_linkage(node)
                color = line_colors.to_rgba(dist)
                ax[0].text(px + tr_label_margin, py,
                           show_tr_label(dist), ha='left', va='center', color=color)

                # Left
                ccnt, cdepth, clcntl, crcntl, clcdist = node_props[child_left]
                # print('left', ccnt, cdepth, clcntl, crcntl)
                # cx = maxheight - cdepth
                cx = tr_left_margin + maxcumdist - clcdist
                cy = (prev_lcnt - crcntl) * ts_height
                if py == cy:
                    cy -= 1 / 2 * ts_height
                # print('plot line', (px, cx), (py, cy))
                # ax[0].add_line(Line2D((px, cx), (py, cy), lw=2, color='black', axes=ax[0]))
                ax[0].add_line(Line2D((px, px), (py, cy), lw=1, color=color, axes=ax[0]))
                ax[0].add_line(Line2D((px, cx), (cy, cy), lw=1, color=color, axes=ax[0]))
                cnt_ts = plot_i(child_left, depth + 1, cnt_ts, prev_lcnt - crcntl, ax, True)

                # Right
                ccnt, cdepth, clcntr, crcntr, crcdist = node_props[child_right]
                # print('right', ccnt, cdepth, clcntr, crcntr)
                # cx = maxheight - cdepth
                cx = tr_left_margin + maxcumdist - crcdist
                cy = (prev_lcnt + clcntr) * ts_height
                if py == cy:
                    cy += 1 / 2 * ts_height
                # print('plot line', (px, cx), (py, cy))
                # ax[0].add_line(Line2D((px, cx), (py, cy), lw=2, color='black', axes=ax[0]))
                ax[0].add_line(Line2D((px, px), (py, cy), lw=1, color=color, axes=ax[0]))
                ax[0].add_line(Line2D((px, cx), (cy, cy), lw=1, color=color, axes=ax[0]))
                cnt_ts = plot_i(child_right, depth + 1, cnt_ts, prev_lcnt + clcntr, ax, False)
            return cnt_ts

        plot_i(self.maxnode, 0, 0, node_props[self.maxnode][2], ax, True)

        if filename:
            if isinstance(filename, Path):
                filename = str(filename)
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
            fig, ax = None, None

        return fig, ax

    def to_dot(self):
        child_left, child_right, dist, cnt = self.get_linkage(self.maxnode)
        node_deque = deque([(self.maxnode, child_left), (self.maxnode, child_right)])
        # print(node_deque)
        s = ["digraph tree {"]
        while len(node_deque) > 0:
            from_node, to_node = node_deque.popleft()
            s.append("  {} -> {};".format(from_node, to_node))
            if to_node >= len(self.series):
                child_left, child_right, dist, cnt = self.get_linkage(to_node)
                node_deque.append((to_node, child_left))
                node_deque.append((to_node, child_right))
            # print(node_deque)
        s.append("}")
        return "\n".join(s)


class HierarchicalTree(BaseTree):
    """Wrapper to keep track of the full tree that represents the hierarchical clustering.

    The linkage tree is available in self.linkage.

    :param model: Clustering object. For example of class :class:`Hierarchical`.
        If no model is given, the arguments are identical to those of class :class:`Hierarchical`.
    """

    def __init__(self, model=None, **kwargs):
        if model is None:
            self._model = Hierarchical(**kwargs)
        else:
            self._model = model
        super().__init__(**kwargs)
        if not math.isinf(self._model.max_dist):
            logger.info('Resetting max_dist to infinity. Otherwise the result is not guaranteed to '
                        'be a single rooted tree and cannot be visualized.')
            self._model.max_dist = float('inf')

    def fit(self, series, *args, **kwargs):
        """Fit a hierarchical clustering tree.

        All arguments are passed when calling the model past to `__init__`.
        The linkage tree is also available in self.linkage.

        :param series: Sequence over time series
        :return: Linkage datastructure
        """
        self.series = SeriesContainer.wrap(series)
        self.linkage = []
        new_nodes = {i: i for i in range(len(series))}
        if self._model.merge_hook:
            old_merge_hook = self._model.merge_hook
        else:
            old_merge_hook = None

        def merge_hook(from_idx, to_idx, distance):
            # print('merge_hook', from_idx, to_idx)
            new_idx = len(self.series) + len(self.linkage)
            # print('adding to linkage: ', new_nodes[from_idx], new_nodes[to_idx], distance, 0)
            if new_nodes[from_idx] is None:
                raise Exception('Trying to merge series that is already merged')
            self.linkage.append((new_nodes[from_idx], new_nodes[to_idx], distance, 0))
            new_nodes[to_idx] = new_idx
            new_nodes[from_idx] = None
            if old_merge_hook:
                old_merge_hook(from_idx, to_idx, distance)

        self._model.merge_hook = merge_hook

        result = self._model.fit(series, *args, **kwargs)
        self._model.merge_hook = old_merge_hook
        return result


class LinkageTree(BaseTree):
    """Hierarchical clustering using the Scipy linkage function.

    The linkage tree is available in self.linkage.

    This is the same but faster algorithm as available in Hierarchical (~10 times faster). But with less
    options to steer the clustering (e.g. no possibility to give weights). It still computes the entire
    distance matrix first and is thus not ideal for extremely large data sets.
    """

    def __init__(self, dists_fun, dists_options=None, method='complete'):
        """

        :param dists_fun: Distance funcion, e.g. dtw.distance
        :param dists_options: Options passed to dists_fun
        :param method: Linkage method (see scipy.cluster.hierarchy.linkage)
        """
        if dists_options is None:
            dists_options = {}
        super().__init__()
        self.dists_fun = dists_fun
        self.dists_options = dists_options
        self.method = method

    def fit(self, series):
        """Fit a hierarchical clustering tree.

        The linkage tree is also available in self.linkage.

        :param series: Sequence over time series
        :return: Linkage datastructure
        """
        if np is None:
            raise NumpyException("The fit function requires Numpy to be installed.")
        try:
            from scipy.cluster.hierarchy import linkage
        except ImportError:
            raise ScipyException("The LinkageTree class requires the scipy package to be installed.")
        self.series = SeriesContainer.wrap(series)
        dists = self.dists_fun(self.series, **self.dists_options)
        dists_cond = np.zeros(self._size_cond(len(series)))
        idx = 0
        for r in range(len(series) - 1):
            dists_cond[idx:idx + len(series) - r - 1] = dists[r, r + 1:]
            idx += len(series) - r - 1

        self.linkage = linkage(dists_cond, method=self.method, metric='euclidean')
        return self.linkage

    def _size_cond(self, size):
        n = int(size)
        return int((n * (n - 1)) / 2)


class Hooks:
    @staticmethod
    def create_weighthook(weights, series):
        def newhook(i1, i2, dist):
            w1 = weights[i1]
            w2 = weights[i2]
            p1 = series[i1]
            p2 = series[i2]
            if w1 < w2 or (w1 == w2 and len(p1) > len(p2)):
                i1, i2 = i2, i1
            weights[i1] = w1 + w2
            return i1, i2
        return newhook

    @staticmethod
    def create_orderhook(weights):
        def newhook(idxs):
            min_idx = -1
            max_weight = -1
            for r, c in [idxs[ii, :] for ii in range(idxs.shape[0])]:
                total = weights[r] + weights[c]
                if total > max_weight:
                    max_weight = total
                    min_idx = (r, c)
            return min_idx
        return newhook
