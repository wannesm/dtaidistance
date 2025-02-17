# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw_visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW) visualisations.

:author: Wannes Meert
:copyright: Copyright 2017 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import os
import logging
import math

from .util import dtaidistance_dir
from .util_numpy import NumpyException

logger = logging.getLogger("be.kuleuven.dtai.distance")

from . import dtw
try:
    from . import dtw_c
except ImportError:
    # logger.info('C library not available')
    dtw_c = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from tqdm import tqdm
except ImportError:
    logger.info('tqdm library not available')
    tqdm = None


def plot_warping(s1, s2, path, filename=None):
    """Plot the optimal warping between two sequences.

    :param s1: From sequence.
    :param s2: To sequence.
    :param path: Optimal warping path.
        Can be computed with methods like ``dtw_ndim.warping_path``.
    :param filename: Filename path (optional).
    """
    if np is None:
        raise NumpyException('Function plot_warping requires Numpy.')
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    ax[0].pcolormesh(np.transpose(s1))
    ax[1].pcolormesh(np.transpose(s2))
    transFigure = fig.transFigure.inverted()
    lines = []
    line_options = {'linewidth': 2, 'color': 'orange', 'alpha': 0.8}
    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        coord1 = transFigure.transform(ax[0].transData.transform([r_c+.5, 0]))
        coord2 = transFigure.transform(ax[1].transData.transform([c_c+.5, 0]))
        lines.append(mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                      transform=fig.transFigure, **line_options))
    fig.lines = lines
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax


def plot_warpingpaths(s1, s2, paths, path=None, filename=None, shownumbers=False):
    """Plot the warping paths matrix.

    :param s1: Series 1
    :param s2: Series 2
    :param paths: Warping paths matrix
    :param path: Path to draw (typically this is the best path)
    :param filename: Filename for the image (optional)
    :param shownumbers: Show distances also as numbers
    """
    from matplotlib import pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(10, 10), frameon=True)
    gs = gridspec.GridSpec(2, 2, wspace=1, hspace=1,
                           left=0, right=1.0, bottom=0, top=1.0,
                           height_ratios=[1, 6],
                           width_ratios=[1, 6])

    if path is None:
        p = dtw.best_path(paths)
    else:
        p = path

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    ax0.text(0, 0, "Dist = {:.4f}".format(paths[p[-1][0], p[-1][1]]))
    ax0.xaxis.set_major_locator(plt.NullLocator())
    ax0.yaxis.set_major_locator(plt.NullLocator())

    # Top time series
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.set_ylim([0, s2.shape[1]])
    ax1.set_axis_off()
    ax1.xaxis.tick_top()
    ax1.pcolormesh(np.transpose(s2))
    ax1.xaxis.set_major_locator(plt.NullLocator())
    ax1.yaxis.set_major_locator(plt.NullLocator())

    # Left time series
    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.set_xlim([0, s1.shape[1]])
    ax2.set_axis_off()
    ax2.xaxis.set_major_locator(plt.NullLocator())
    ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.pcolormesh(np.flipud(s1))

    ax3 = fig.add_subplot(gs[1:, 1:])
    ax3.matshow(paths[1:, 1:])
    py, px = zip(*p)
    ax3.plot(px, py, ".-", color="red")
    if shownumbers:
        for r in range(1, paths.shape[0]):
            for c in range(1, paths.shape[1]):
                ax3.text(c - 1, r - 1, "{:.2f}".format(paths[r, c]))

    gs.tight_layout(fig, pad=1.0, h_pad=1.0, w_pad=1.0)

    # Align the subplots:
    ax1pos = ax1.get_position().bounds
    ax2pos = ax2.get_position().bounds
    ax3pos = ax3.get_position().bounds
    ax2.set_position((ax2pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax2pos[2], ax3pos[3])) # adjust the subplot on the left vertically
    if len(s1) < len(s2):
        ax3.set_position((ax3pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax3pos[2], ax3pos[3])) # move the subplot on the left and the distance matrix upwards
    if len(s1) > len(s2):
        ax3.set_position((ax1pos[0], ax3pos[1], ax3pos[2], ax3pos[3])) # move the subplot at the top and the distance matrix to the left
        ax1.set_position((ax1pos[0], ax1pos[1], ax3pos[2], ax1pos[3])) # adjust the subplot at the top horizontally
    if len(s1) == len(s2):
        ax1.set_position((ax3pos[0], ax1pos[1], ax3pos[2], ax1pos[3])) # adjust the subplot at the top horizontally

    ax = fig.axes

    if filename:
        if type(filename) != str:
            filename = str(filename)
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax