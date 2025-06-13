import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import dtaidistance.dtw_visualisation as dtwvis
from dtaidistance import innerdistance
from dtaidistance import util_numpy
from dtaidistance.dtw import warping_path, warping_paths, best_path
from dtaidistance.exceptions import MatplotlibException
from dtaidistance.explain.dsw.explainpair import ExplainPair, SplitStrategy, rdp_vectorized
from dtaidistance.preprocessing import smoothing

# If environment variable TESTDIR is set, save figures to this
directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
logger = logging.getLogger("be.kuleuven.dtai.distance")
numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")
scipyonly = pytest.mark.skipif("util_numpy.test_without_scipy()")


@numpyonly
def test_diffbaseline():
    from dtaidistance.benchmarks.synthetic import pattern1

    with util_numpy.test_uses_numpy() as np:
        ys = []
        x = np.linspace(0, 30, num=20)
        x00 = 7
        x1 = 22
        params = [
            (0, 0.5, 1),
            (-0.3, 0.5, 1),
            (-0.1, 0.5, 2),
            (0.1, 0.5, 2),
            (0.2, 0.5, 3)
        ]
        for x0d, c, d in params:
            y = pattern1(x, x00 + x0d, c, x1, d)
            y = y[:10]
            ys.append(y)
        ya, yb = ys[0], ys[4]
        ya = ya * 2.0
        yb = yb + 5.0
        d, paths = warping_paths(ya, yb)
        print(paths)
        path = best_path(paths)
        print(path)
        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")
            fig, axs = dtwvis.plot_warping(ya, yb, path)
            fn = directory / "test_diffbaseline.png"
            print(f"Saving to {fn}")
            fig.savefig(str(fn))
            plt.close()


@scipyonly
@numpyonly
def test_explain1():
    from dtaidistance.benchmarks.synthetic import pattern1
    with util_numpy.test_uses_numpy() as np:
        ys = []
        x = np.linspace(0, 30, num=200)
        c = 0.5
        x00 = 7
        x1 = 22

        params = [
            (0, 0.5, 1),
            (-0.3, 0.5, 1),
            (-0.1, 0.5, 2),
            (0.1, 0.5, 2),
            (0.2, 0.5, 3)
        ]

        for x0d, c, d in params:
            y = pattern1(x, x00 + x0d, c, x1, d)
            ys.append(y)

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")

            # Plot all series
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 4),
                                   sharex='all', sharey='all')
            fn = directory / "test_explain1_series.png"
            ax[0].plot(ys[0])
            for y in ys[1:]:
                ax[1].plot(y)
            fig.savefig(str(fn))
            plt.close()


def prepare_sin_wave_ts(length_of_ts, starting_index_of_wave, length_of_wave):
    return np.concatenate((np.zeros(starting_index_of_wave), np.sin(np.linspace(0, 2 * np.pi, length_of_wave)),
                           np.zeros(length_of_ts - starting_index_of_wave - length_of_wave)))


@numpyonly
def test_ssm():
    from dtaidistance.benchmarks.synthetic import pattern1
    with util_numpy.test_uses_numpy() as np:
        ys = []
        x = np.linspace(0, 30, num=200)
        x00 = 7
        x1 = 22
        r = 0.05  # 0.0 # 0.05  # 0.1

        delta_rel = 1
        delta_abs = 0.3518

        approx_prune = True

        params = [
            (0, 0.5, 1., 1),
            (-0.7, 0.5, 1., 1),
            (-0.3, 0.5, 2., 2),
            (2.0, 0.55, 1., 2),
            (0.4, 0.5, 1., 3),
            (2.0, 0.5, 1., 2)]

        for x0d, c, a, d in params:
            y = pattern1(x, x00 + x0d, c, a, x1, d, r=r, rs=3980)
            ys.append(y)
        ya = ys[0]
        yb = ys[3]
        dist, paths = warping_paths(ya, yb)

        path = best_path(paths)

        ep = ExplainPair(ya, yb, delta_rel=delta_rel, delta_abs=delta_abs,
                         approx_prune=approx_prune)
        inner_dist, inner_res, inner_val = innerdistance.inner_dist_fns(ep.dtw_settings.inner_dist)

        path3 = ep.line2
        dist_approx3 = ep.distance_approx()

        Delta_rel = inner_res(inner_val(dist) * delta_rel) / dist
        Delta_abs = inner_res(inner_val(dist) + delta_abs) - dist
        dist_approx3_max = dist * (1 + Delta_rel) + Delta_abs
        assert dist_approx3 <= dist_approx3_max
        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")

            print(str(directory / "test_ccm_path_pair.png"))
            ep.plot_warping(str(directory / "test_ccm_path_pair.png"))

            fig, axs = dtwvis.plot_warpingpaths(ya, yb, paths, path=path)
            # axs[0].plot(path2[:, 1], path2[:, 0], '-', alpha=0.5, color='yellow', linewidth=3)
            axs[0].plot(path3[:, 1], path3[:, 0], '-o', alpha=0.5, color='cyan', linewidth=3)
            axs[3].text(0, 0.2, f"Dist_a = {dist_approx3:.4f} < {dist_approx3_max:.4f}")
            # axs[3].text(0, 0.4, f"Dist(i) = {dist_approx2:.4f}")

            rdist, rdists = ep.distance(per_segment=True)
            adist, adists = ep.distance_approx(per_segment=True)
            for segment, srdist, sadist in zip(ep.segments, rdists, adists):
                axs[0].text(segment.s_idx_y + 4, segment.s_idx,
                            f"{segment.s_idx_p} - {srdist:.4f}/{sadist:.4f} - "
                            f"({segment.s_idx},{segment.s_idx_y})", color="red")
            print('')
            print(rdists)
            print(f"{rdist=} ?= inner_res({sum(rdists)}) = {inner_res(sum(rdists))}")
            print(adists)
            print(f"{adist=} ?= inner_res({sum(adists)}) = {inner_res(sum(adists))}")
            fn = directory / "test_ccm_paths_1.png"
            fig.savefig(str(fn))
            plt.close(fig)


@numpyonly
def test_ssm2():
    with util_numpy.test_uses_numpy() as np:
        rsrc_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'rsrc', 'Trace_TRAIN.txt')
        print(rsrc_fn)
        data = np.loadtxt(rsrc_fn)
        labels = data[:, 0]
        series = data[:, 1:]
        seriesf = series[labels == 1]
        seriesf = smoothing(seriesf, smooth=0.1)

        ya = seriesf[12, :]
        yb = seriesf[1, :]

        dist, paths = warping_paths(ya, yb)
        path = best_path(paths)

        ep2 = ExplainPair(ya, yb, delta_rel=2, delta_abs=1, approx_prune=True)
        path2 = ep2.line2
        dist_approx2 = ep2.distance_approx()

        ep3 = ExplainPair(ya, yb, delta_rel=7, delta_abs=1, approx_prune=True)
        path3 = ep3.line2
        dist_approx3 = ep3.distance_approx()

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")

            fig, axs = dtwvis.plot_warpingpaths(ya, yb, paths, path=path)
            axs[0].plot(path2[:, 1], path2[:, 0], '-', alpha=0.5, color='yellow', linewidth=3)
            axs[0].plot(path3[:, 1], path3[:, 0], '-o', alpha=0.5, color='cyan', linewidth=3)
            axs[3].text(0, 0.4, f"Dist(d) = {dist_approx3:.4f}")
            axs[3].text(0, 0.2, f"Dist(i) = {dist_approx2:.4f}")

            rdist, rdists = ep3.distance(per_segment=True)
            adist, adists = ep3.distance_approx(per_segment=True)
            for segment, srdist, sadist in zip(ep3.segments, rdists, adists):
                axs[0].text(segment.s_idx_y + 4, segment.s_idx,
                            f"{srdist:.4f} / {sadist:.4f} - "
                            f"({segment.s_idx},{segment.s_idx_y})", color="red")

            fn = directory / "test_ssm2_paths.png"
            fig.savefig(str(fn))
            plt.close(fig)


# The use cases in test_ssm3() are not working as expected. The assertation is commented as for now.
def test_ssm3():
    ya = prepare_sin_wave_ts(120, 10, 100)
    yb = prepare_sin_wave_ts(120, 30, 60)

    # configuration for ExplainPair
    delta_rel = 2
    delta_abs = 0.1
    ep = ExplainPair(ya, yb, delta_rel=delta_rel, delta_abs=delta_abs,
                     approx_prune=True)

    dist, paths = warping_paths(ya, yb)
    inner_dist, inner_res, inner_val = innerdistance.inner_dist_fns(ep.dtw_settings.inner_dist)
    Delta_rel = inner_res(inner_val(dist) * delta_rel) / dist
    Delta_abs = inner_res(inner_val(dist) + delta_abs) - dist
    dist_approx3_max = dist * (1 + Delta_rel) + Delta_abs
    dist_approx3 = ep.distance_approx()
    assert dist_approx3 <= dist_approx3_max


@numpyonly
def test_explain_pair():
    from dtaidistance.benchmarks.synthetic import pattern1
    with util_numpy.test_uses_numpy() as np:
        ys = []
        x = np.linspace(0, 30, num=200)
        c = 0.5
        x00 = 7
        x1 = 22

        params = [
            (0, 0.5, 1., 1),
            (-0.7, 0.5, 1., 1),
            (-0.3, 0.5, 2., 2),
            (0.7, 0.5, 1., 2),
            (0.4, 0.5, 1., 3)
        ]

        for x0d, c, a, d in params:
            y = pattern1(x, x00 + x0d, c, a, x1, d)
            ys.append(y)
        ya = ys[0]
        yb = ys[3]

        ep = ExplainPair(ya, yb, delta_rel=2, delta_abs=0.1, approx_prune=True, onlychanges=2)

        if directory and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise MatplotlibException("No matplotlib available")

            # Plot warping
            fig, axs = plt.subplots(nrows=4, ncols=1, sharey=True, sharex=True, figsize=(10, 10))
            path = warping_path(ya, yb)
            dtwvis.plot_warping(ya, yb, path, fig=fig, axs=[axs[0], axs[1]])

            ep.plot_warping(fig=fig, axs=[axs[2], axs[3]])

            fn = directory / "test explain_pair_warping.png"
            fig.savefig(str(fn))
            plt.close(fig)


def test_plot_rdp():
    line = np.array([[24, 173], [26, 170], [24, 166], [27, 162], [37, 161], [45, 157], [48, 152],
                     [46, 143], [40, 140], [34, 137], [26, 134], [24, 130], [24, 125], [28, 121],
                     [36, 118], [46, 117], [63, 121], [76, 125], [82, 120], [86, 111], [88, 103],
                     [90, 91], [95, 87], [107, 89], [107, 104], [106, 117], [109, 129], [119, 131],
                     [131, 131], [139, 134], [138, 143], [131, 152], [119, 154], [111, 149],
                     [105, 143], [91, 139], [80, 142], [81, 152], [76, 163], [67, 161], [59, 149], [63, 138]])
    plt.plot(line[:, 0], line[:, 1], '-o');
    line2, _ = rdp_vectorized(line, 8.8)
    plt.plot(line2[:, 0], line2[:, 1], '-o');
    plt.show()


def test_sine_pathdiff():
    ya = prepare_sin_wave_ts(120, 10, 100)
    yb = prepare_sin_wave_ts(120, 30, 60)

    ep = ExplainPair(ya, yb,
                     delta_rel=1,
                     delta_abs=0.1,
                     approx_prune=False,
                     # split_strategy="spatialdist",
                     split_strategy=SplitStrategy.PATH_DIFF,
                     )

    if directory is not None:
        ep.plot_warping(filename=str(directory / "test_sine_pathdiff.png"))
        ep.plot_segments(filename=str(directory / "test_sine_pathdiff_segments.png"))


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print("Saving files to {}".format(directory))
    test_ssm()
