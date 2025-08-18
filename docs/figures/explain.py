import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dtaidistance.dtw import warping_path, warping_paths, best_path
from dtaidistance.explain.dsw.explainpair import ExplainPair, SplitStrategy
from dtaidistance.benchmarks.synthetic import pattern1, pattern2
from dtaidistance.dtw_visualisation import plot_warping, plot_warpingpaths


here = Path(__file__).parent
subplots_kw = dict(nrows=2, ncols=1, figsize=(5,3), sharex=True, sharey=True)
subplots_kw2 = dict(nrows=2, ncols=2, figsize=(2*5,3), sharex=True, sharey=True)


def generate_pattern1():
    x = np.linspace(0, 30, num=200)

    x00 = 7
    x1 = 22
    r = 0.03
    rs = 3980  # random seed
    params = [
        (0, 0.5, 1., 1),
        (2.0, 0.5, 1.3, 2),
        (2.0, 0.55, 1., 2),
    ]

    x2 = 22
    params2 = [
        (0, 0.5, 1., 2),
        (6.0, 0.5, 1., 4)
    ]
    ys = [pattern1(x, x00 + x0d, c, a, x1, d, r=r, rs=rs) for x0d, c, a, d in params]
    ys2 = [pattern1(x, x00 + x0d, c, a, x2, d, r=r, rs=rs) for x0d, c, a, d in params2]
    return ys + ys2


def example1():
    ys = generate_pattern1()
    ya, yb = ys[0], ys[1]

    fn = str(here / "_build" / "explain_example1_dtw.png")
    fig, axs = plt.subplots(**subplots_kw2)
    dist, paths = warping_paths(ya, yb)
    print(f"{dist=}")  # 0.25
    path = best_path(paths)
    # plot_warping(ya, yb, path, filename=fn, fig=fig, axs=axs)
    plot_warping(ya, yb, path, fig=fig, axs=axs[:,0])
    plot_warping(ya, yb, path, fig=fig, axs=axs[:,1], start_on_curve=False, color_misalignment=True)
    fig.savefig(fn)
    plt.close(fig)


    fn = str(here / "_build" / "explain_example1_dsw.png")
    fig, axs = plt.subplots(**subplots_kw)
    ep = ExplainPair(ya, yb, delta_rel=2, delta_abs=0.5, split_strategy=SplitStrategy.DERIV_DIST)
    ep.plot_warping(filename=fn, fig=fig, axs=axs)


def example2():
    with open(str(here/ "data" / "data_from_ucr.json"), "r") as json_file:
        ucr_data = json.load(json_file)
    data = ucr_data["ECG"]
    ya = np.array(data["ts1"])
    yb = np.array(data["ts2"])

    fn = str(here / "_build" / "explain_example2_dtw.png")
    fig, axs = plt.subplots(**subplots_kw2)
    dist, paths = warping_paths(ya, yb)
    print(f"{dist=}")  # 6.07
    path = best_path(paths)
    plot_warping(ya, yb, path, fig=fig, axs=axs[:,0])
    plot_warping(ya, yb, path, fig=fig, axs=axs[:,1], start_on_curve=False, color_misalignment=True)
    fig.savefig(fn)
    plt.close(fig)

    fn = str(here / "_build" / "explain_example2_dsw.png")
    fig, axs = plt.subplots(**subplots_kw)
    ep = ExplainPair(ya, yb, delta_rel=1, delta_abs=2.0, split_strategy=SplitStrategy.DERIV_DIST)
    ep.plot_warping(filename=fn, fig=fig, axs=axs)


example1()
example2()

