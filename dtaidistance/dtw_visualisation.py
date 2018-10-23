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
import numpy as np

from .util import dtaidistance_dir

logger = logging.getLogger("be.kuleuven.dtai.distance")

from . import dtw
try:
    from . import dtw_c
except ImportError:
    # logger.info('C library not available')
    dtw_c = None

try:
    from tqdm import tqdm
except ImportError:
    logger.info('tqdm library not available')
    tqdm = None


def plot_warp(from_s, to_s, new_s, path, filename=None):
    """Plot the warped sequence and its relation to the original sequence
    and the target sequence.

    :param from_s: From sequence.
    :param to_s: To sequence.
    :param new_s: Warped version of from sequence.
    :param path: Optimal warping path.
    :param filename: Filename path (optional).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        logger.error("The plot_warp function requires the matplotlib package to be installed.")
        return
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    ax[0].plot(from_s, label="From")
    ax[0].legend()
    ax[1].plot(to_s, label="To")
    ax[1].legend()
    transFigure = fig.transFigure.inverted()
    lines = []
    line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        coord1 = transFigure.transform(ax[0].transData.transform([r_c, from_s[r_c]]))
        coord2 = transFigure.transform(ax[1].transData.transform([c_c, to_s[c_c]]))
        lines.append(mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                      transform=fig.transFigure, **line_options))
    ax[2].plot(new_s, label="From-warped")
    ax[2].legend()
    for i in range(len(to_s)):
        coord1 = transFigure.transform(ax[1].transData.transform([i, to_s[i]]))
        coord2 = transFigure.transform(ax[2].transData.transform([i, new_s[i]]))
        lines.append(mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                      transform=fig.transFigure, **line_options))
    fig.lines = lines
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax


def plot_warping(s1, s2, path, filename=None):
    """Plot the optimal warping between to sequences.

    :param s1: From sequence.
    :param s2: To sequence.
    :param path: Optimal warping path.
    :param filename: Filename path (optional).
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    ax[0].plot(s1)
    ax[1].plot(s2)
    transFigure = fig.transFigure.inverted()
    lines = []
    line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        coord1 = transFigure.transform(ax[0].transData.transform([r_c, s1[r_c]]))
        coord2 = transFigure.transform(ax[1].transData.transform([c_c, s2[c_c]]))
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
    from matplotlib.ticker import FuncFormatter

    ratio = max(len(s1), len(s2))
    min_y = min(np.min(s1), np.min(s2))
    max_y = max(np.max(s1), np.max(s2))

    fig = plt.figure(figsize=(10, 10), frameon=True)
    gs = gridspec.GridSpec(2, 2, wspace=1, hspace=1,
                           left=0, right=1.0, bottom=0, top=1.0,
                           height_ratios=[1, 6],
                           width_ratios=[1, 6])
    max_s2_x = np.max(s2)
    max_s2_y = len(s2)
    max_s1_x = np.max(s1)
    min_s1_x = np.min(s1)
    max_s1_y = len(s1)

    if path is None:
        p = dtw.best_path(paths)
    else:
        p = path

    def format_fn2_x(tick_val, tick_pos):
        return max_s2_x - tick_val

    def format_fn2_y(tick_val, tick_pos):
        return int(max_s2_y - tick_val)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    ax0.text(0, 0, "Dist = {:.4f}".format(paths[p[-1][0], p[-1][1]]))
    ax0.xaxis.set_major_locator(plt.NullLocator())
    ax0.yaxis.set_major_locator(plt.NullLocator())

    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.set_ylim([min_y, max_y])
    ax1.set_axis_off()
    ax1.xaxis.tick_top()
    # ax1.set_aspect(0.454)
    ax1.plot(range(len(s2)), s2, ".-")
    ax1.xaxis.set_major_locator(plt.NullLocator())
    ax1.yaxis.set_major_locator(plt.NullLocator())

    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.set_xlim([-max_y, -min_y])
    ax2.set_axis_off()
    # ax2.set_aspect(0.8)
    # ax2.xaxis.set_major_formatter(FuncFormatter(format_fn2_x))
    # ax2.yaxis.set_major_formatter(FuncFormatter(format_fn2_y))
    ax2.xaxis.set_major_locator(plt.NullLocator())
    ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.plot(-s1, range(max_s1_y, 0, -1), ".-")

    ax3 = fig.add_subplot(gs[1:, 1:])
    # ax3.set_aspect(1)
    ax3.matshow(paths[1:, 1:])
    # ax3.grid(which='major', color='w', linestyle='-', linewidth=0)
    # ax3.set_axis_off()
    py, px = zip(*p)
    ax3.plot(px, py, ".-", color="red")
    # ax3.xaxis.set_major_locator(plt.NullLocator())
    # ax3.yaxis.set_major_locator(plt.NullLocator())
    if shownumbers:
        for r in range(1, paths.shape[0]):
            for c in range(1, paths.shape[1]):
                ax3.text(c - 1, r - 1, "{:.2f}".format(paths[r, c]))

    gs.tight_layout(fig, pad=1.0, h_pad=1.0, w_pad=1.0)
    # fig.subplots_adjust(hspace=0, wspace=0)

    ax = fig.axes

    if filename:
        if type(filename) != str:
            filename = str(filename)
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax

def plot_matrix(distances, filename=None, ax=None, shownumbers=False):
    from matplotlib import pyplot as plt

    if ax is None:
        if shownumbers:
            figsize = (15, 15)
        else:
            figsize = None
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else:
        fig = None

    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('both')

    im = ax.imshow(distances)
    idxs = [str(i) for i in range(len(distances))]
    # Show all ticks
    ax.set_xticks(np.arange(len(idxs)))
    ax.set_xticklabels(idxs)
    ax.set_yticks(np.arange(len(idxs)))
    ax.set_yticklabels(idxs)

    ax.set_title("Distances between series", pad=30)

    if shownumbers:
        for i in range(len(idxs)):
            for j in range(len(idxs)):
                if not np.isinf(distances[i, j]):
                    l = "{:.2f}".format(distances[i, j])
                    ax.text(j, i, l, ha="center", va="center", color="w")

    if filename:
        if type(filename) != str:
            filename = str(filename)
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax
