import math
import numpy as np

from ..util import SeriesContainer
from ..subsequence.subsequencealignment import subsequence_alignment
from ..exceptions import MatplotlibException
from ..similarity import distance_to_similarity


class SymbolAlignment:
    def __init__(self, codebook, maxcompression=0.5, maxexpansion=2):
        """Translate a time series with continuous values to a list of discrete
        symbols based on motifs in a codebook.

        :param codebook: List of motifs.
        :param maxcompression: Maximally allowed compression of a codeword for it to be recognized
        :param maxexpansion: Maximally allowed expansion of a codeword for it to be recognized
        """
        self.codebook = codebook
        self.maxcompression = maxcompression
        self.maxexpansion = maxexpansion
        self.use_c = False
        self.symbols = None
        self._agg_fn = self.agg_min
        self._agg_args = None

    def set_agg_min(self):
        """Set mininum aggregation for the align2 method."""
        self._agg_fn = self.agg_min
        self._agg_args = None

    def agg_min(self, patterns, max_value):
        return np.argmin(patterns, axis=2).astype(int)

    def set_agg_prob(self, window=10):
        """Set probabilistic aggregation for the align2 method."""
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

    def align2(self, series):
        """Perform alignment.

        For each time point, select the best matching motif and aggregate the results.

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
                for match in sa.kbest_matches(k=None,
                                              minlength=math.floor(len(medoidd)*self.maxcompression),
                                              maxlength=math.ceil(len(medoidd)*self.maxexpansion)):
                    patterns[sidx, match.segment[0]:match.segment[1]+1, midx] = match.value
                    max_value = max(max_value, match.value)
        patterns[:, :, len(self.codebook)] = 0
        patterns[:, :, len(self.codebook)] = np.max(patterns) + 1
        best_patterns = self._agg_fn(patterns, max_value)
        self.symbols = best_patterns
        return best_patterns

    def align(self, series, max_rangefactor=None, detectknee_alpha=None, max_overlap=None):
        """Perform alignment.

        Only one motif is matched to a subsequence. No aggregation within the same subsequence.

        :param series: List of time series or a numpy array
        :param max_rangefactor: the range between the first (best) match and the last match
            can be at most a factor of ``maxrangefactor``. For example, if the first match has
            value v_f, then the last match has a value ``v_l < v_f*maxfactorrange``.
        :param max_overlap: Maximal overlap when matching codewords.
            If not given, this is based on maxcompression and maxexpansion.
        """
        # Inspired on the Matching Pursuit algorithm.
        sc = SeriesContainer(series)
        noword = len(self.codebook)
        best_patterns = np.full(series.shape, noword, dtype=int)
        if max_overlap is None:
            max_overlap = max(self.maxcompression, 1./self.maxexpansion)

        for sidx in range(len(sc)):
            curseries = sc[sidx].copy()
            patterns = []
            max_value = 0
            for midx in range(len(self.codebook)):
                medoidd = np.array(self.codebook[midx])
                sa = subsequence_alignment(medoidd, curseries, use_c=self.use_c)
                if max_rangefactor is not None:
                    itr = sa.best_matches(max_rangefactor=max_rangefactor,
                                          minlength=math.floor(len(medoidd) * self.maxcompression),
                                          maxlength=math.ceil(len(medoidd) * self.maxexpansion))
                else:
                    itr = sa.best_matches_knee(alpha=detectknee_alpha,
                                          minlength=math.floor(len(medoidd) * self.maxcompression),
                                          maxlength=math.ceil(len(medoidd) * self.maxexpansion))
                for match in itr:
                    patterns.append((midx, match.segment[0], match.segment[1]+1,
                                     curseries[match.segment[0]:match.segment[1]+1], match.value))
                    max_value = max(max_value, match.value)
            # print(f"Series {sidx}: found {len(patterns)} patterns")
            # D = np.zeros((len(patterns), len(curseries)))
            # for i, (_, bi, ei, ts) in enumerate(patterns):
            #     D[i, bi:ei] = ts # should be normalized for matching pursuit if we use the factors
            D = np.zeros(len(patterns))
            L = np.zeros(len(patterns), dtype=int)
            B = np.zeros(len(patterns), dtype=int)
            E = np.zeros(len(patterns), dtype=int)
            for i, (_, bi, ei, ts, v) in enumerate(patterns):
                D[i] = v
                L[i] = ei - bi + 1
                B[i] = bi
                E[i] = ei + 1  # Use next position for length and overlap computation
            # Trade-off between length and similarity
            S = distance_to_similarity(D, r=max_value, method='exponential') * L

            bestpatvalue = np.inf
            its = 0
            while bestpatvalue > 0:
                # print(f"Iteration {sidx} -- {its}")
                # dotprod = np.einsum('j,kj->k', curseries, D)
                # bestpatidx = np.argmax(dotprod)
                # bestpatvalue = dotprod[bestpatidx]
                bestpatidx = np.argmax(S)
                bestpatvalue = S[bestpatidx]
                if bestpatvalue == 0:
                    break
                pattern = patterns[bestpatidx]
                freeidxs = best_patterns[sidx, pattern[1]:pattern[2]] == noword
                best_patterns[sidx, pattern[1]:pattern[2]][freeidxs] = pattern[0]
                # Remove patterns with more overlap than max_overlap
                overlaps = (np.maximum(0, np.minimum(E[bestpatidx], E) -
                                          np.maximum(B[bestpatidx], B)) / L[bestpatidx]) > max_overlap
                S[overlaps] = 0
                # curseries[pattern[1]:pattern[2]] = 0
                # D[bestpatidx, :] = 0
                S[bestpatidx] = 0
                its += 1

        self.symbols = best_patterns
        return best_patterns

    def align_fast(self, *args, **kwargs):
        """See :meth:`align`."""
        use_c = self.use_c
        self.use_c = True
        result = self.align(*args, **kwargs)
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

    def plot(self, series, sequences, sequences_idx, ylabels=None, filename=None, figsize=None,
             xlimits=None, symbollabels=None):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
        except ImportError:
            raise MatplotlibException("No matplotlib available")
        if figsize is None:
            figsize = (12, 8)
        sc = SeriesContainer(series)
        fig, axs = plt.subplots(nrows=len(sc), ncols=1, sharex='all', sharey='col', figsize=figsize)
        if len(sc) == 1:
            axs = [axs]
        for r in range(series.shape[0]):
            # avg_value = series[r, :].mean()
            avg_value = series.min() + (series.max() - series.min()) / 2
            if xlimits is None:
                axs[r].plot(series[r, :])
            else:
                axs[r].plot(series[r, xlimits[0]:xlimits[1]])
            if ylabels is not None:
                axs[r].set_ylabel(f"L={ylabels[r]}")
            for symbol, (fidx, lidx) in zip(sequences[r], sequences_idx[r]):
                if xlimits is None:
                    delta = 0
                elif xlimits[0] <= fidx <= xlimits[1]:
                    delta = xlimits[0]
                else:
                    continue
                axs[r].vlines(fidx-delta, series.min(), series.max(), colors='k', alpha=0.2)
                if symbollabels is not None:
                    symbol = symbollabels(symbol)
                axs[r].text(fidx-delta, avg_value, str(symbol), alpha=0.5)
        if filename is not None:
            fig.savefig(filename)
            plt.close(fig)
        else:
            return fig, axs
