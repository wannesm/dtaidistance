import os
import ast
from pathlib import Path

import pytest

from dtaidistance import util_numpy
from dtaidistance.symbolization.alignment import SymbolAlignment
from dtaidistance.preprocessing import differencing
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance.exceptions import MatplotlibException


# If environment variable TESTDIR is set, save figures to this
# directory, otherwise do not save figures
directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))

numpyonly = pytest.mark.skipif("util_numpy.test_without_numpy()")


@numpyonly
def test_trace1():
    print(f"Using directory: {directory}")
    with util_numpy.test_uses_numpy() as np:
        # Load data
        rsrc = Path(__file__).parent / 'rsrc'
        rsrc_fn = rsrc / 'Trace_TRAIN.txt'
        data = np.loadtxt(rsrc_fn)
        labels = data[:, 0]
        series = data[:, 1:]

        # Filter data (to speed up test)
        nb_occurrences = 3  # occurrence of each class
        data2 = np.zeros((4 * nb_occurrences, data.shape[1]))
        cnts = [0] * (4 + 1)
        for r in range(data.shape[0]):
            label = int(data[r, 0])
            if cnts[label] < nb_occurrences:
                data2[cnts[label] + (label - 1) * nb_occurrences, :] = data[r, :]
                cnts[label] += 1
        data = data2
        print(f"Data: {data.shape}")
        data = data[np.argsort(data[:, 0])]
        labels = data[:, 0]
        series = data[:, 1:]

        # Load motifs (learned with LoCoMotif)
        with (rsrc / 'trace_motifs.py').open('r') as fp:
            data = fp.read()
        array = np.array
        data = ast.literal_eval(data)
        medoidsd = data["medoidd"]
        medoids = data["medoid"]

        # Plot motifs
        if directory is not None and not dtwvis.test_without_visualization():
            try:
                import matplotlib.pyplot as plt
                from matplotlib import gridspec
            except ImportError:
                raise MatplotlibException("No matplotlib available")

            fig, axs = plt.subplots(nrows=len(medoids), ncols=2, sharex=True, sharey="col", figsize=(6, 8))
            fig.subplots_adjust(wspace=0.4, hspace=1)
            for idx, (medoid, medoidd) in enumerate(zip(medoids, medoidsd)):
                axs[idx, 0].plot(medoid)
                axs[idx, 0].set_title(f"Medoid {idx + 1}")
                axs[idx, 1].plot(medoidd)
            fig.savefig(directory / "medoids.png")
            plt.close(fig)

        # Preprocess time series
        seriesd = differencing(series, smooth=0.1)

        # Symbolization
        sa = SymbolAlignment(codebook=medoidsd)
        symbols = sa.align_fast(seriesd)
        if directory is not None:
            np.savetxt(str(directory / "symbolized.npy"), symbols, fmt='%i')

        # Hangover
        sequences, sequences_idx = sa.hangover(symbols, threshold=4)
        if directory is not None and not dtwvis.test_without_visualization():
            sa.plot(series, sequences, sequences_idx,
                    ylabels=labels, filename=directory / "series.png")
            sa.plot(seriesd, sequences, sequences_idx,
                    ylabels=labels, filename=directory / "seriesd.png")
