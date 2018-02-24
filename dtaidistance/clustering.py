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
import math
from pathlib import Path
from collections import defaultdict, namedtuple, deque
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger("be.kuleuven.dtai.distance")


class Hierarchical:
    def __init__(self, dists_fun, dists_options, max_dist=np.inf,
                 merge_hook=None, order_hook=None, show_progress=True):
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
        """
        self.dists_fun = dists_fun
        self.dists_options = dists_options
        self.max_dist = max_dist
        self.merge_hook = merge_hook
        self.order_hook = order_hook
        self.show_progress = show_progress

    def fit(self, series):
        """Merge sequences.

        :param series: Iterator over series.
        :return: Dictionary with as keys the prototype indicices and as values all the indicides of the series in
            that cluster.
        """
        nb_series = len(series)
        cluster_idx = dict()
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
        while min_value <= self.max_dist:
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
                    cluster_idx[i] = set(i)
        return cluster_idx


class BaseTree:

    def __init__(self):
        """

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html:
        A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are
        combined to form cluster n + i. A cluster with an index less than n corresponds to one of the original
        observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value
        Z[i, 3] represents the number of original observations in the newly formed cluster.
        """
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

    def plot(self, filename=None, axes=None):
        """Plot the hierarchy and time series.

        :param filename: If a filename is passed, the image is written to this file.
        :param axes: If a axes array is passed the image is added to this figure.
            Expects axes[0] and axes[1] to be present.
        """
        # print('linkage')
        # for l in self.linkage:
        #     print(l)
        self._series_y = [0] * len(self.series)
        ts_height = 10
        ts_bottom_margin = 2
        ts_top_margin = 2
        tr_unit = 1

        max_dist = 0
        for _, _, d, _ in self.linkage:
            if not np.isinf(d):
                max_dist = max(max_dist, d)

        node_props = dict()

        max_y = max(np.max(self.series), abs(np.min(self.series)))
        self.ts_height_factor = (ts_height / max_y) * 0.9

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
            ax = axes
        ax[0].set_axis_off()
        # ax[0].set_xlim(left=0, right=curdept)
        ax[0].set_xlim(left=0, right=maxcumdist + 0.05)
        ax[0].set_ylim(bottom=0, top=tr_unit * len(self.series))
        # ax[0].plot([0,1],[1,2])
        # ax[0].add_line(Line2D((0,1),(2,2), lw=2, color='black', axes=ax[0]))

        ax[1].set_axis_off()
        ax[1].set_xlim(left=0, right=len(self.series[0]))
        ax[1].set_ylim(bottom=0, top=ts_bottom_margin + ts_height*len(self.series) + ts_top_margin)

        cnt_ts = 0

        def plot_i(node, depth, cnt_ts, prev_lcnt, ax):
            # print('plot_i', node, depth, cnt_ts, prev_lcnt)
            pcnt, pdepth, plcnt, prcnt, pcdist = node_props[node]
            # px = maxheight - pdepth
            px = maxcumdist - pcdist
            py = prev_lcnt * tr_unit
            if node < len(self.series):
                # Plot series
                # print('plot series y={}'.format(ts_bottom_margin + ts_height * cnt_ts + self.ts_height_factor))
                self._series_y[int(node)] = ts_bottom_margin + ts_height * cnt_ts
                serie = self.series[int(node)]
                ax[1].text(0, ts_bottom_margin + ts_height * cnt_ts + self.ts_height_factor, int(node))
                ax[1].plot(ts_bottom_margin + ts_height * cnt_ts + self.ts_height_factor * serie)
                cnt_ts += 1

            else:
                child_left, child_right, dist, _ = self.get_linkage(node)
                ax[0].text(px + 0.05, py + tr_unit / 10, "{:.2f}".format(dist))

                # Left
                ccnt, cdepth, clcntl, crcntl, clcdist = node_props[child_left]
                # print('left', ccnt, cdepth, clcntl, crcntl)
                # cx = maxheight - cdepth
                cx = maxcumdist - clcdist
                cy = (prev_lcnt - crcntl) * tr_unit
                if py == cy:
                    cy -= 1/2*tr_unit
                # print('plot line', (px, cx), (py, cy))
                # ax[0].add_line(Line2D((px, cx), (py, cy), lw=2, color='black', axes=ax[0]))
                ax[0].add_line(Line2D((px, px), (py, cy), lw=1, color='black', axes=ax[0]))
                ax[0].add_line(Line2D((px, cx), (cy, cy), lw=1, color='black', axes=ax[0]))
                cnt_ts = plot_i(child_left, depth + 1, cnt_ts, prev_lcnt - crcntl, ax)

                # Right
                ccnt, cdepth, clcntr, crcntr, crcdist = node_props[child_right]
                # print('right', ccnt, cdepth, clcntr, crcntr)
                # cx = maxheight - cdepth
                cx = maxcumdist - crcdist
                cy = (prev_lcnt + clcntr) * tr_unit
                if py == cy:
                    cy += 1/2*tr_unit
                # print('plot line', (px, cx), (py, cy))
                # ax[0].add_line(Line2D((px, cx), (py, cy), lw=2, color='black', axes=ax[0]))
                ax[0].add_line(Line2D((px, px), (py, cy), lw=1, color='black', axes=ax[0]))
                ax[0].add_line(Line2D((px, cx), (cy, cy), lw=1, color='black', axes=ax[0]))
                cnt_ts = plot_i(child_right, depth + 1, cnt_ts, prev_lcnt + clcntr, ax)
            return cnt_ts

        plot_i(self.maxnode, 0, 0, node_props[self.maxnode][2], ax)

        if filename:
            if isinstance(filename, Path):
                filename = str(filename)
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

        return ax

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
    def __init__(self, model, *args, **kwargs):
        """Wrapper to keep track of the full tree that represents the hierarchical clustering.

        :param model: Clustering object. For example of type `Hierarchical`.
        """
        self._model = model
        super().__init__(*args, **kwargs)
        self._model.max_dist = np.inf

    def fit(self, series, *args, **kwargs):
        if isinstance(series, np.matrix):
            self.series = np.asarray(series)
        else:
            self.series = series
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
    def __init__(self, dists_fun, dists_options):
        """Hierarchical clustering using the Scipy linkage function.

        This is the same but faster algorithm as available in Hierarchical (~10 times faster). But with less
        options to steer the clustering (e.g. no possibility to give weights). It still computes the entire
        distance matrix first and is thus not ideal for extremely large data sets.
        """
        super().__init__()
        self.dists_fun = dists_fun
        self.dists_options = dists_options

    def fit(self, series):
        from scipy.cluster.hierarchy import linkage
        if isinstance(series, np.matrix):
            self.series = np.asarray(series)
        else:
            self.series = series
        dists = self.dists_fun(series, **self.dists_options)
        dists_cond = np.zeros(self._size_cond(len(series)))
        idx = 0
        for r in range(len(series) - 1):
            dists_cond[idx:idx + len(series) - r - 1] = dists[r, r + 1:]
            idx += len(series) - r - 1

        self.linkage = linkage(dists_cond, method='complete', metric='euclidean')

    def _size_cond(self, size):
        n = int(size)
        # r = 2
        # f = math.factorial
        # return int(f(n) / f(r) / f(n - r))
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
