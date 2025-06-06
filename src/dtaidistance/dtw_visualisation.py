# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw_visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW) visualisations.

:author: Wannes Meert
:copyright: Copyright 2017-2024 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import os
import logging

from . import util_numpy


try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
except ImportError:
    np = None


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


def test_without_visualization():
    if "DTAIDISTANCE_TESTWITHOUTVIZ" in os.environ and os.environ["DTAIDISTANCE_TESTWITHOUTVIZ"] == "1":
        return True
    return False


def plot_warp(from_s, to_s, new_s, path, filename=None, fig=None, axs=None):
    """Plot the warped sequence and its relation to the original sequence
    and the target sequence.

    :param from_s: From sequence.
    :param to_s: To sequence.
    :param new_s: Warped version of from sequence.
    :param path: Optimal warping path.
    :param filename: Filename path (optional).
    :param fig: Matplotlib Figure object
    :param axs: Array of Matplotlib axes.Axes objects (length == 3)
    :return: Figure, list[Axes]
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.patches import ConnectionPatch
    except ImportError:
        logger.error("The plot_warp function requires the matplotlib package to be installed.")
        return
    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex='all', sharey='all')
    elif fig is None or axs is None:
        raise TypeError(f'The fig and axs arguments need to be both None or both instantiated.')
    axs[0].plot(from_s, label="From")
    axs[0].legend()
    axs[1].plot(to_s, label="To")
    axs[1].legend()
    lines = []
    line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        con = ConnectionPatch(xyA=[r_c, from_s[r_c]], coordsA=axs[0].transData,
                              xyB=[c_c, to_s[c_c]], coordsB=axs[1].transData, **line_options)

        lines.append(con)
    axs[2].plot(new_s, label="From-warped")
    axs[2].legend()
    for i in range(len(to_s)):
        con = ConnectionPatch(xyA=[i, to_s[i]], coordsA=axs[1].transData,
                              xyB=[i, new_s[i]], coordsB=axs[2].transData, **line_options)
        lines.append(con)
    for line in lines:
        fig.add_artist(line)
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, axs = None, None
    return fig, axs


def plot_warping(s1, s2, path, filename=None, fig=None, axs=None,
                 series_line_options=None, warping_line_options=None):
    """Plot the optimal warping between two sequences.

    :param s1: From sequence.
    :param s2: To sequence.
    :param path: Optimal warping path.
        Can be computed with methods like :meth:`dtaidistance.dtw.warping_path`.
    :param filename: Filename path (optional).
    :param fig: Matplotlib Figure object
    :param axs: Array of Matplotlib axes.Axes objects (length == 2)
    :param series_line_options: Dictionary of options to pass to matplotlib plot
        None will not pass any options
    :param warping_line_options: Dictionary of options to pass to matplotlib ConnectionPatch
        None will use {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    :return: Figure, list[Axes]
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.patches import ConnectionPatch
    except ImportError:
        logger.error("The plot_warp function requires the matplotlib package to be installed.")
        return
    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
    elif fig is None or axs is None:
        raise TypeError(f'The fig and axs arguments need to be both None or both instantiated.')
    if series_line_options is None:
        series_line_options = {}
    axs[0].plot(s1, **series_line_options)
    axs[1].plot(s2, **series_line_options)
    plt.tight_layout()
    lines = []
    if warping_line_options is None:
        warping_line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        con = ConnectionPatch(xyA=[r_c, s1[r_c]], coordsA=axs[0].transData,
                              xyB=[c_c, s2[c_c]], coordsB=axs[1].transData, **warping_line_options)
        lines.append(con)
    for line in lines:
        fig.add_artist(line)
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, axs = None, None
    return fig, axs


def plot_warping_single_ax(s1, s2, path, filename=None, fig=None, ax=None):
    """Plot the optimal warping between to sequences.

    :param s1: From sequence.
    :param s2: To sequence.
    :param path: Optimal warping path.
    :param filename: Filename path (optional).
    :param fig: Matplotlib Figure object
    :param ax: Matplotlib axes.Axes object
    :return: Figure, Axes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.patches import ConnectionPatch
    except ImportError:
        logger.error("The plot_warp function requires the matplotlib package to be installed.")
        return
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif fig is None or ax is None:
        raise TypeError(f'The fig and ax arguments need to be both None or both instantiated.')
    ax.plot(s1)
    ax.plot(s2)
    lines = []
    line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        con = ConnectionPatch(xyA=[r_c, s1[r_c]], coordsA=ax.transData,
                              xyB=[c_c, s2[c_c]], coordsB=ax.transData, **line_options)
        lines.append(con)
    for line in lines:
        fig.add_artist(line)
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax


def path_slice(path, rb=None, re=None, cb=None, ce=None):
    path2 = []
    for t in path:
        if rb is not None and t[0] < rb:
            continue
        if cb is not None and t[1] < cb:
            continue
        if re is not None and t[0] > (re - 1):
            continue
        if ce is not None and t[1] > (ce - 1):
            continue
        path2.append((t[0] - rb, t[1] - cb))
    return path2


def plot_warpingpaths(s1, s2, paths, cost_matrix = None, path=None, filename=None,
                      shownumbers=False, showlegend=False, showtotaldist=True,
                      figure=None, path_kwargs = None, matshow_kwargs=None, includes_zero=True, tick_kwargs=None):
    """Plot the warping paths matrix.

    :param s1: Series 1
    :param s2: Series 2
    :param paths: Warping paths matrix
    :param cost_matrix: Cost matrix, if it is not None, it will be plotted instead of the accummulated cost matrix ('paths').
    :param path: Path to draw (typically this is the best path)
    :param filename: Filename for the image (optional)
    :param shownumbers: Show distances also as numbers
    :param showlegend: Show colormap legend
    :param figure: Matplotlib Figure object
    :param path_kwargs: kwargs for the path plot
    :param matshow_kwargs: kwargs for the matshow plot

    :return: Figure, Axes
    """
    try:
        from matplotlib import pyplot as plt
        from matplotlib import gridspec
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        logger.error("The plot_warpingpaths function requires the matplotlib package to be installed.")
        return
    ratio = max(len(s1), len(s2))
    min_y = min(np.min(s1), np.min(s2))
    max_y = max(np.max(s1), np.max(s2))

    if figure is None:
        fig = plt.figure(figsize=(10, 10), frameon=True)
    else:
        fig = figure
    if showlegend:
        grows = 3
        gcols = 3
        height_ratios = [1, 6, 1]
        width_ratios = [1, 6, 1]
    else:
        grows = 2
        gcols = 2
        height_ratios = [1, 6]
        width_ratios = [1, 6]
    gs = gridspec.GridSpec(grows, gcols, wspace=1, hspace=1,
                           left=0, right=1.0, bottom=0, top=1.0,
                           height_ratios=height_ratios,
                           width_ratios=width_ratios)
    max_s2_x = np.max(s2)
    max_s2_y = len(s2) - 1
    max_s1_y = len(s1) - 1
    y_ratio2 = (np.max(s1) - np.min(s1)) / (np.max(s2) - np.min(s2))
    y_ratio1 = min(1.0, 1.0 / y_ratio2)
    y_ratio2 = min(1.0, y_ratio2)

    if path is None and includes_zero is True:
        p = dtw.best_path(paths)
    elif type(path) is int and path == -1:
        p = None
    else:
        p = path

    def format_fn2_x(tick_val, tick_pos):
        return max_s2_x - tick_val

    def format_fn2_y(tick_val, tick_pos):
        return int(max_s2_y - tick_val)

    # Warping path
    ax3 = fig.add_subplot(gs[1, 1])
    # ax3.set_aspect(1)
    if matshow_kwargs is None:
        matshow_kwargs = {}
    if cost_matrix is not None:
        img = ax3.matshow(cost_matrix[1:, 1:], aspect='equal', **matshow_kwargs)
    else:
        if includes_zero:
            img = ax3.matshow(paths[1:, 1:], aspect='equal', **matshow_kwargs)
        else:
            img = ax3.matshow(paths, aspect='equal', **matshow_kwargs)
    # ax3.grid(which='major', color='w', linestyle='-', linewidth=0)
    # ax3.set_axis_off()
    if p is not None:
        if type(p) is list:
           py, px = zip(*p)
        else:
            py = p[:, 0]
            px = p[:, 1]

        path_kwargs = {'color':"red"} if path_kwargs is None else path_kwargs
        ax3.plot(px, py, ".-", **path_kwargs)

    # ax3.xaxis.set_major_locator(plt.NullLocator())
    # ax3.yaxis.set_major_locator(plt.NullLocator())
    if shownumbers:
        for r in range(1, paths.shape[0]):
            for c in range(1, paths.shape[1]):
                ax3.text(c - 1, r - 1, "{:.2f}".format(paths[r, c]), ha='center', va='center')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('right')

    # Time series on top axis
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_ylim([min_y, max_y])
    ax1.set_axis_off()
    ax1.xaxis.tick_top()
    # ax1.set_aspect(0.454)

    ax1.plot(range(len(s2)), s2, ".-", color = '#ff7f0e')
    ax1.set_xlim([-0.5, len(s2) - 0.5])
    ax1.xaxis.set_major_locator(plt.NullLocator())
    ax1.yaxis.set_major_locator(plt.NullLocator())

    # Time series on left axis
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim([-max_y, -min_y])
    ax2.set_axis_off()
    # ax2.set_aspect(0.8)
    # ax2.xaxis.set_major_formatter(FuncFormatter(format_fn2_x))
    # ax2.yaxis.set_major_formatter(FuncFormatter(format_fn2_y))
    ax2.xaxis.set_major_locator(plt.NullLocator())
    ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.plot(-s1, range(0, max_s1_y + 1), ".-", color = '#1f77b4')
    ax2.set_ylim([-0.5, max_s1_y + 0.5])
    ax2.invert_yaxis()

    # for ax in [ax0, ax1, ax2, ax3]:
    #     for spine in ax.spines.values():
    #         spine.set_visible(True)
    #         spine.set_edgecolor('green')  # Set border color
    #         spine.set_linewidth(2)  # Set border thickness

    gs.tight_layout(fig, pad=1.0, h_pad=1.0, w_pad=1.0)
    # fig.subplots_adjust(hspace=0, wspace=0)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    if p is not None and showtotaldist:
        ax0.text(0, 0, "Dist = {:.4f}".format(paths[p[-1][0] + 1, p[-1][1] + 1]))
    ax0.xaxis.set_major_locator(plt.NullLocator())
    ax0.yaxis.set_major_locator(plt.NullLocator())

    if showlegend:
        # ax4 = fig.add_subplot(gs[0:, 2])
        ax4 = fig.add_axes([0.9, 0.25, 0.015, 0.5])
        fig.colorbar(img, cax=ax4)

    # Align the subplots:
    if len(s1) != len(s2):
        # bounds = (xmin, ymin, width, height)
        ax1pos = ax1.get_position().bounds  # top ts
        ax2pos = ax2.get_position().bounds  # left ts
        ax3pos = ax3.get_position().bounds  # warping path
        left = ax3pos[0]
        bottom = ax3pos[1]
        width = ax3pos[2]
        height = ax3pos[3]
        dist_between_y = ax1pos[1] - (ax2pos[1] + ax2pos[3])
        dist_between_x = ax2pos[0] - (ax1pos[0])
        # set_position([left, bottom, width, height])
        ax1.set_position((left, bottom+height+dist_between_y, width, ax1pos[3]))
        ax2.set_position((left + dist_between_x, bottom, ax2pos[2], height))


    ax = fig.axes
    if tick_kwargs is not None:
        ax3.tick_params(**tick_kwargs)
    # ax3.spines['bottom'].set_linewidth(2.0)
    # ax3.spines['right'].set_linewidth(2.0)
    for spine in ax3.spines.values():
        spine.set_linewidth(2)  # Increase axis thickness
        # spine.set_color('black')  # Ensure it's visible
    if filename:
        if type(filename) != str:
            filename = str(filename)
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax


def plot_warpingpaths_addpath(ax, path):
    py, px = zip(*path)
    ax3 = ax[3]
    ax3.plot(px, py, ".-", color="red", markersize=2)


def plot_matrix(distances, shownumbers=False, filename=None, fig=None, ax=None):
    from matplotlib import pyplot as plt

    if ax is None and fig is None:
        if shownumbers:
            figsize = (15, 15)
        else:
            figsize = None
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    elif fig is None or ax is None:
        raise TypeError(f'The fig and ax arguments need to be both None or both instantiated.')
    else:
        fig = None

    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('both')

    im = ax.imshow(distances)
    idxs_y = [str(i) for i in range(distances.shape[0])]
    idxs_x = [str(i) for i in range(distances.shape[1])]
    # Show all ticks
    ax.set_xticks(np.arange(len(idxs_x)))
    ax.set_xticklabels(idxs_x)
    ax.set_yticks(np.arange(len(idxs_y)))
    ax.set_yticklabels(idxs_y)

    ax.set_title("Distances between series", pad=30)

    if shownumbers:
        for i in range(len(idxs_y)):
            for j in range(len(idxs_x)):
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


def plot_average(s1, s2, avg, path1, path2, filename=None, fig=None, ax=None):
    """Plot how s1 and s2 relate to the avg.

    :param s1: Seq 1.
    :param s2: Seq 2.
    :param path: Average sequence.
    :param filename: Filename path (optional).
    :param fig: Matplotlib Figure object
    :param ax: Matplotlib axes.Axes object
    :return: Figure, axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.patches import ConnectionPatch
    except ImportError:
        logger.error("The plot_warp function requires the matplotlib package to be installed.")
        return
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all', sharey='all')
    elif fig is None or ax is None:
        raise TypeError(f'The fig and axs arguments need to be both None or both instantiated.')
    ax.plot(s1, color='blue', alpha=0.5)
    ax.plot(s2, color='blue', alpha=0.5)
    ax.plot(avg, color='orange', linestyle='dashed', alpha=0.5)
    # plt.tight_layout()
    # lines = []
    # line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    # for r_c, c_c in path:
    #     if r_c < 0 or c_c < 0:
    #         continue
    #     con = ConnectionPatch(xyA=[r_c, s1[r_c]], coordsA=ax[0].transData,
    #                           xyB=[c_c, s2[c_c]], coordsB=ax[1].transData, **line_options)
    #     lines.append(con)
    # for line in lines:
    #     fig.add_artist(line)
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax
