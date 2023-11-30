import numpy as np

from ..util import SeriesContainer
from ..subsequence.subsequencealignment import subsequence_alignment
from ..exceptions import MatplotlibException


class SymbolAlignment:
    def __init__(self, codebook):
        """Translate a time series with continuous values to a list of discrete
        symbols based on motifs in a codebook.

        :param codebook: List of motifs.
        """
        self.codebook = codebook
        self.use_c = False
        self.symbols = None

    def align(self, series):
        """Perform alignment.

        :param series: List of time series or a numpy array
        """
        sc = SeriesContainer(series)

        patterns = np.zeros((len(sc), sc.get_max_length(), len(self.codebook) + 1))
        patterns[:, :, :] = np.inf
        for sidx in range(len(sc)):
            for midx in range(len(self.codebook)):
                medoidd = np.array(self.codebook[midx])
                sa = subsequence_alignment(medoidd, sc[sidx], use_c=self.use_c)
                for match in sa.kbest_matches(k=None):
                    patterns[sidx, match.segment[0]:match.segment[1], midx] = match.value
        patterns[:, :, len(self.codebook)] = 0
        patterns[:, :, len(self.codebook)] = np.max(patterns) + 1
        best_patterns = np.argmin(patterns, axis=2).astype(int)
        self.symbols = best_patterns
        return best_patterns

    def align_fast(self, series):
        use_c = self.use_c
        self.use_c = True
        result = self.align(series)
        self.use_c = use_c
        return result

    def hangover(self, symbols, threshold=4):
        """Hangover filter for symbols."""
        sequences = []
        sequences_idx = []
        for r in range(symbols.shape[0]):
            sequence = []
            sequence_idx = []
            lastval = None
            lastcnt = 0
            firstidx = None
            lastsaved = None
            for c, v in enumerate(symbols[r, :]):
                if v != lastval:
                    if lastcnt > threshold and lastval != lastsaved:
                        sequence.append(lastval + 1)  # cannot be zero
                        sequence_idx.append((firstidx, c))
                        lastsaved = lastval
                    lastval = v
                    lastcnt = 0
                    firstidx = c
                else:
                    lastcnt += 1
            sequences.append(sequence)
            sequences_idx.append(sequence_idx)
        return sequences, sequences_idx

    def plot(self, series, sequences, sequences_idx, labels=None, filename=None):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except ImportError:
            raise MatplotlibException("No matplotlib available")
        sc = SeriesContainer(series)
        fig, axs = plt.subplots(nrows=len(sc), ncols=1, sharex=True, sharey="col", figsize=(12, 8))
        for r in range(series.shape[0]):
            axs[r].plot(series[r, :])
            if labels is not None:
                axs[r].set_ylabel(f"L={labels[r]}")
            for symbol, (fidx, lidx) in zip(sequences[r], sequences_idx[r]):
                axs[r].text(fidx, 0, str(symbol))
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig, axs
