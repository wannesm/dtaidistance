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
    def __init__(self, dists_fun, dists_options, max_dist=np.inf, weights=None, merge_hook=None, show_progress=True):
        """

        :param dists_fun: Function to compute pairwise distance matrix between set of series.
        :param dists_options: Arguments to pass to dists_fun.
        :param max_dist: Do not merge or cluster series that are further apart than this.
        :param weights: Weight per series. Used to pick the prototype that defines a cluster.
            The clustering will try to group such that the prototypes have a high as possible summed weight.
        :param merge_hook: Function that is called when two series are clustered.
            The function definition is `def merge_hook(from_idx, to_idx, distance)`, where idx is the index of the series.
        :param show_progress: Use a tqdm progress bar
        """
        self.dists_fun = dists_fun
        self.dists_options = dists_options
        self.weights = weights
        self.max_dist = max_dist
        self.merge_hook = merge_hook
        self.show_progress = show_progress

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
        logger.debug('Merging patterns')
        if self.show_progress and tqdm:
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
                self.weights[i1] = w1 + w2
            logger.debug("Merge {} <- {} ({:.3f})".format(i1, i2, min_value))
            if i1 not in cluster_idx:
                cluster_idx[i1] = {i1}
            if i2 in cluster_idx:
                cluster_idx[i1].update(cluster_idx[i2])
                del cluster_idx[i2]
            else:
                cluster_idx[i1].add(i2)

            if self.merge_hook:
                self.merge_hook(i2, i1, min_value)
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
            if np.isinf(min_value):
                break
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


class HierarchicalTree(Hierarchical):
    def __init__(self, *args, **kwargs):
        """Keep track of the full tree that represents the hierarchical clustering."""
        super().__init__(*args, **kwargs)
        self.max_dist = np.inf
        self.children_left = None
        self.children_right = None
        self.series = None
        self.node_props = None

    def fit(self, series, *args, **kwargs):
        self.series = series
        self.children_left = [-1 for i in range(len(series) + 1)]
        self.children_right = [-1 for i in range(len(series) + 1)]
        self.node_props = dict()
        new_nodes = {i: i+1 for i in range(len(series))}
        if self.merge_hook:
            old_merge_hook = self.merge_hook
        else:
            old_merge_hook = None

        def merge_hook(from_idx, to_idx, distance):
            self.children_left.append(new_nodes[from_idx])
            self.children_right.append(new_nodes[to_idx])
            new_idx = len(self.children_left) - 1
            self.node_props[new_idx] = distance
            new_nodes[to_idx] = new_idx
            new_nodes[from_idx] = None
            if old_merge_hook:
                old_merge_hook(from_idx, to_idx, distance)

        self.merge_hook = merge_hook

        result = super().fit(series, *args, **kwargs)
        self.node_props[0] = self.node_props[len(self.children_left) - 1]
        self.children_left[0] = self.children_left[-1]
        self.children_left.pop()
        self.children_right[0] = self.children_right[-1]
        self.children_right.pop()

        # print(self.children_left)
        # print(self.children_right)
        return result

    def to_dot(self):
        node_deque = deque([(0, self.children_left[0]), (0,self.children_right[0])])
        s = ["digraph tree {"]
        while len(node_deque) > 0:
            from_node, to_node = node_deque.popleft()
            s.append("  {} -> {};".format(from_node, to_node))
            if self.children_left[to_node] < 0:
                pass
            else:
                node_deque.append((to_node, self.children_left[to_node]))
            if self.children_right[to_node] < 0:
                pass
            else:
                node_deque.append((to_node, self.children_right[to_node]))
            # print(node_deque)
        s.append("}")
        return "\n".join(s)

    def plot(self, filename=None):
        ts_height = 10
        ts_bottom_margin = 2
        ts_top_margin = 2
        tr_unit = 1

        node_props = dict()

        max_y = max(np.max(self.series), abs(np.min(self.series)))
        ts_height_factor = ts_height / max_y

        def count(node, height):
            maxheight = None
            curdepth = None
            cnt = 0
            left_cnt = None
            right_cnt = None
            if self.children_left[node] < 0:
                cnt += 1
                maxheight = height
                curdepth = 0
                left_cnt = 0
                right_cnt = 0
            else:
                nc, nmh, ncd = count(self.children_left[node], height + 1)
                cnt += nc
                maxheight = nmh
                curdepth = ncd + 1
                left_cnt = nc
            if self.children_right[node] < 0:
                # c += 1
                pass
            else:
                nc, nmh, ncd = count(self.children_right[node], height + 1)
                cnt += nc
                maxheight = max(maxheight, nmh)
                curdepth = max(curdepth, ncd + 1)
                right_cnt = nc
            # print('c', node, c)
            node_props[node] = (cnt, curdepth, left_cnt, right_cnt)
            return cnt, maxheight, curdepth

        cnt, maxheight, curdepth = count(0, 0)

        fig, ax = plt.subplots(nrows=1, ncols=2, frameon=False)
        ax[0].set_axis_off()
        ax[0].set_xlim(left=0, right=maxheight)
        ax[0].set_ylim(bottom=0, top=tr_unit * len(self.series))
        # ax[0].plot([0,1],[1,2])
        # ax[0].add_line(Line2D((0,1),(2,2), lw=2, color='black', axes=ax[0]))

        ax[1].set_axis_off()
        ax[1].set_xlim(left=0, right=len(self.series[0]))
        ax[1].set_ylim(bottom=0, top=ts_bottom_margin + ts_height*len(self.series) + ts_top_margin)

        cnt_ts = 0

        def plot_i(node, depth, cnt_ts, prev_lcnt, ax):
            # print('plot_i', node, depth, cnt_ts, prev_lcnt)
            pcnt, pdepth, plcnt, prcnt = node_props[node]
            px = maxheight - pdepth
            py = prev_lcnt * tr_unit
            if node in self.node_props:
                ax[0].text(px, py + tr_unit/10, self.node_props[node])
            if self.children_left[node] < 0:
                # Plot series
                # print('plot series')
                ax[1].plot(ts_bottom_margin + ts_height * cnt_ts + ts_height_factor * self.series[node - 1])
                cnt_ts += 1
            else:
                ccnt, cdepth, clcntl, crcntl = node_props[self.children_left[node]]
                # print('left', ccnt, cdepth, clcntl, crcntl)
                cx = maxheight - cdepth
                cy = (prev_lcnt - crcntl) * tr_unit
                if py == cy:
                    cy -= 1/2*tr_unit
                # print('plot line', (px, cx), (py, cy))
                ax[0].add_line(Line2D((px, cx), (py, cy), lw=2, color='black', axes=ax[0]))
                cnt_ts = plot_i(self.children_left[node], depth + 1, cnt_ts, prev_lcnt - crcntl, ax)
            if self.children_right[node] < 0:
                # print('plot ', node - 1)
                # ax[1].plot(tsheight * (1 + cnt_ts) + self.series[node - 1])
                pass
            else:
                ccnt, cdepth, clcntr, crcntr = node_props[self.children_right[node]]
                # print('right', ccnt, cdepth, clcntr, crcntr)
                cx = maxheight - cdepth
                cy = (prev_lcnt + clcntr) * tr_unit
                if py == cy:
                    cy += 1/2*tr_unit
                # print('plot line', (px, cx), (py, cy))
                ax[0].add_line(Line2D((px, cx), (py, cy), lw=2, color='black', axes=ax[0]))
                cnt_ts = plot_i(self.children_right[node], depth + 1, cnt_ts, prev_lcnt + clcntr, ax)
            return cnt_ts

        plot_i(0, 0, 0, node_props[0][2], ax)

        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        else:
            plt.show(block=True)
