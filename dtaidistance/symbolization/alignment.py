import numpy as np

from ..util import SeriesContainer
from ..subsequence.subsequencealignment import subsequence_alignment
from ..exceptions import MatplotlibException
from ..similarity import distance_to_similarity


class SymbolAlignment:
    def __init__(self, codebook):
        """Translate a time series with continuous values to a list of discrete
        symbols based on motifs in a codebook.

        :param codebook: List of motifs.
        """
        self.codebook = codebook
        self.use_c = False
        self.symbols = None
        self._agg_fn = self.agg_min
        self._agg_args = None

    def set_agg_min(self):
        self._agg_fn = self.agg_min
        self._agg_args = None

    def agg_min(self, patterns, max_value):
        return np.argmin(patterns, axis=2).astype(int)

    def set_agg_prob(self, window=10):
        self._agg_fn = self.agg_prob
        self._agg_args = (window,)

    def agg_prob(self, patterns, max_value):
        window = self._agg_args[0]
        tlen = patterns.shape[1]
        patterns = distance_to_similarity(patterns, r=max_value, method='exponential')
        # Softmax
        exp_x = np.exp(patterns - np.max(patterns, axis=2)[:, :, np.newaxis])
        logprobs = np.log(exp_x / np.sum(exp_x, axis=2)[:, :, np.newaxis])
        best = np.zeros((patterns.shape[0], tlen), dtype=int)
        for ti in range(window):
            best[:, ti] = np.argmax(logprobs[:, :ti, :].sum(axis=1), axis=1)
        for ti in range(window, tlen-window):
            best[:, ti] = np.argmax(logprobs[:, ti:ti+window, :].sum(axis=1), axis=1)
        for ti in range(tlen-window, tlen):
            best[:, ti] = np.argmax(logprobs[:, ti:, :].sum(axis=1), axis=1)
        return best

    def align(self, series):
        """Perform alignment.

        :param series: List of time series or a numpy array
        """
        sc = SeriesContainer(series)

        patterns = np.zeros((len(sc), sc.get_max_length(), len(self.codebook) + 1))
        patterns[:, :, :] = np.inf
        max_value = 0
        for sidx in range(len(sc)):
            for midx in range(len(self.codebook)):
                medoidd = np.array(self.codebook[midx])
                sa = subsequence_alignment(medoidd, sc[sidx], use_c=self.use_c)
                for match in sa.kbest_matches(k=None):
                    patterns[sidx, match.segment[0]:match.segment[1], midx] = match.value
                    max_value = max(max_value, match.value)
        patterns[:, :, len(self.codebook)] = 0
        print(f"{np.max(patterns)=}")
        patterns[:, :, len(self.codebook)] = np.max(patterns) + 1
        best_patterns = self._agg_fn(patterns, max_value)
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

    def plot(self, series, sequences, sequences_idx, labels=None, filename=None, figsize=None):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except ImportError:
            raise MatplotlibException("No matplotlib available")
        if figsize is None:
            figsize = (12, 8)
        sc = SeriesContainer(series)
        fig, axs = plt.subplots(nrows=len(sc), ncols=1, sharex=True, sharey="col", figsize=figsize)
        for r in range(series.shape[0]):
            # avg_value = series[r, :].mean()
            avg_value = series.min() + (series.max() - series.min()) / 2
            axs[r].plot(series[r, :])
            if labels is not None:
                axs[r].set_ylabel(f"L={labels[r]}")
            for symbol, (fidx, lidx) in zip(sequences[r], sequences_idx[r]):
                axs[r].vlines(fidx, series.min(), series.max(), colors='k', alpha=0.2)
                axs[r].text(fidx, avg_value, str(symbol), alpha=0.5)
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig, axs
