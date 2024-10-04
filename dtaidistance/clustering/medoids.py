# -*- coding: UTF-8 -*-
"""
dtaidistance.clustering.medoids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Time series clustering using medoid-based methods.

:author: Wannes Meert
:copyright: Copyright 2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import random
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from ..util import SeriesContainer
from ..exceptions import NumpyException, PyClusteringException, MatplotlibException
from .visualization import prepare_plot_options
from ..dtw import DTWSettings


logger = logging.getLogger("be.kuleuven.dtai.distance")


class Medoids:
    def __init__(self, dists_fun, dists_options, k, show_progress=True):
        """

        :param dists_fun:
        :param dists_options:
        :param show_progress:
        """
        self.dists_fun = dists_fun
        self.dists_options = dists_options
        self.show_progress = show_progress
        self.k = k
        self.series = None
        self.cluster_idx = None

    def plot(self, filename=None, axes=None, ts_height=10,
             bottom_margin=2, top_margin=2, ts_left_margin=0, ts_sample_length=1,
             tr_label_margin=3, tr_left_margin=2, ts_label_margin=0,
             show_ts_label=None, show_tr_label=None,
             cmap='viridis_r', ts_color=None):

        try:
            from matplotlib import pyplot as plt
            from matplotlib.lines import Line2D
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
        except ImportError:
            raise MatplotlibException("The plot function requires Matplotlib to be installed.")

        show_ts_label, show_tr_label = prepare_plot_options(show_ts_label, show_tr_label)
        self._series_y = [0] * len(self.series)
        max_y = self.series.get_max_y()
        self.ts_height_factor = (ts_height / max_y) * 0.9

        if axes is None:
            fig, ax = plt.subplots(nrows=1, ncols=2, frameon=False)
        else:
            fig, ax = None, axes
        max_length = max(len(s) for s in self.series)
        ax[0].set_axis_off()
        ax[0].set_xlim(left=0, right=tr_left_margin + ts_sample_length * max_length)
        ax[0].set_ylim(bottom=0, top=bottom_margin + ts_height * len(self.series) + top_margin)
        ax[1].set_axis_off()
        ax[1].set_xlim(left=0, right=ts_left_margin + ts_sample_length * max_length)
        ax[1].set_ylim(bottom=0, top=bottom_margin + ts_height * len(self.series) + top_margin)

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        else:
            pass
        cluster_colors = cmx.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=self.k), cmap=cmap)

        cnt_ts = 0
        for medoid_idx, medoid_id in enumerate(self.cluster_idx.keys()):
            for node in self.cluster_idx[medoid_id]:
                self._series_y[int(node)] = bottom_margin + ts_height * cnt_ts
                serie = self.series[int(node)]

                if ts_color:
                    curcolor = ts_color(int(node))
                else:
                    curcolor = cluster_colors.to_rgba(medoid_idx)

                if node == medoid_id:
                    ax[0].plot(ts_left_margin + ts_sample_length * np.arange(len(serie)),
                               bottom_margin + ts_height * cnt_ts + self.ts_height_factor * serie,
                               color=curcolor)

                ax[1].text(ts_left_margin + ts_label_margin,
                           bottom_margin + ts_height * cnt_ts + ts_height / 2,
                           show_ts_label(int(node)), ha='left', va='center')

                ax[1].plot(ts_left_margin + ts_sample_length * np.arange(len(serie)),
                           bottom_margin + ts_height * cnt_ts + self.ts_height_factor * serie,
                           color=curcolor)
                cnt_ts += 1

        if filename:
            if isinstance(filename, Path):
                filename = str(filename)
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
            fig, ax = None, None

        return fig, ax


class KMedoids(Medoids):
    """KMedoids using the PyClustering package.

    Novikov, A., 2019. PyClustering: Data Mining Library. Journal of Open Source Software, 4(36), p.1230.
    Available at: http://dx.doi.org/10.21105/joss.01230.

    https://pyclustering.github.io/docs/0.9.0/html/d0/dd3/classpyclustering_1_1cluster_1_1kmedoids_1_1kmedoids.html

    """
    def __init__(self, dists_fun, dists_options, k=None, initial_medoids=None, show_progress=True):
        dists_options['compact'] = False
        self.initial_medoids = initial_medoids
        if k is None:
            if initial_medoids is None:
                raise AttributeError('Both k and initial_medoids cannot be None')
            k = len(initial_medoids)
        elif initial_medoids is not None and k != len(initial_medoids):
            raise AttributeError('The length of initial_medoids and k should be identical (or one of the two None)')
        super().__init__(dists_fun, dists_options, k, show_progress)

    def fit(self, series):
        try:
            from pyclustering.cluster.kmedoids import kmedoids
            from pyclustering.utils import calculate_distance_matrix
        except ImportError:
            raise PyClusteringException("The fit function requires the PyClustering package to be installed.")
        if np is None:
            raise NumpyException("The fit function requires Numpy to be installed.")
        self.series = SeriesContainer.wrap(series)
        logger.debug(f'KMedoid: Compute distances between {len(self.series)} series')
        dists = self.dists_fun(self.series, **self.dists_options)
        # Make the matrix symmetric
        i_lower = np.tril_indices(len(self.series), -1)
        dists[i_lower] = dists.T[i_lower]
        dists[np.isinf(dists)] = 0
        if self.initial_medoids is None:
            self.initial_medoids = random.sample(range(len(series)), k=self.k)
        kmedoids_instance = kmedoids(dists, self.initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()
        self.cluster_idx = {medoid: {inst for inst in instances} for medoid, instances in zip(medoids, clusters)}
        return self.cluster_idx
