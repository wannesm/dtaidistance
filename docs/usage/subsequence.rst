Subsequence
-----------

Subsequence search is to match the best occurance of a short time serise in a longer series.

DTW subsequence alignment
~~~~~~~~~~~~~~~~~~~~~~~~~

Given a series:

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/subsequence_series.png?v=1
   :alt: Subsequence series

And a query:

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/subsequence_query.png?v=1
   :alt: Subsequence query

We can find the best occurence(s) as follows:

::

    from dtaidistance.subsequence.dtw import subsequence_alignment
    from dtaidistance import dtw_visualisation as dtwvis

    sa = subsequence_alignment(query, series)
    match = sa.best_match()
    startidx, endidx = match.segment
    dtwvis.plot_warpingpaths(query, series, sa.warping_paths(), match.path, figure=fig)

The resultig match is

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/subsequence_matching.png?v=1
   :alt: Subsequence matching

If we compare the best match with the query we see they are similar.
The best match is only a little bit compressed.

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/subsequence_bestmatch.png?v=1
   :alt: Subsequence best match

If you want to find all matches (or the k best):

::

    fig, ax = dtwvis.plot_warpingpaths(query, series, sa.warping_paths(), path=-1)
    for kmatch in sa.kbest_matches(9):
        dtwvis.plot_warpingpaths_addpath(ax, kmatch.path)


.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/subsequence_bestmatches.png?v=1
   :alt: Subsequence alignment k-best matches


DTW subsequence search (KNN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to using alignment, we can also iterate over a sequence of series or windows
to search for the best match, or best k matches (k-Nearest Neighbors):

::

    from dtaidistance.subsequence.dtw import subsequence_search

    k = 3
    s = []
    w = 22
    ws = int(np.floor(w/2))
    wn = int(np.floor((len(series) - (w - ws)) / ws))
    si, ei = 0, w
    for i in range(wn):
        s.append(series[si:ei])
        si += ws
        ei += ws

    sa = subsequence_search(query, s)
    best = sa.kbest_matches(k=k)

When setting k, the search is pruned to early abandon comparisons
that will not improve on the top k best matches.

In the result one can observe that the choice of windows has an impact
on where the best matches are found. Whereas the previous alignment method
does not require a window size  or a shift, here matches are limited to the windows
that are given. The advantage of this method is that it can be used also if
the windows are not from one continuous series (e.g. periods with missing data,
multiple sources).

The best three windows are visualized below. The gray vertical lines indicate
the windows, the red verical lines the three best windows.


.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/subsequencesearch_bestmatches.png?v=1
   :alt: Subsequence search k-best matches


DTW Local Concurrences
~~~~~~~~~~~~~~~~~~~~~~

In some case we are not interested in searching for a query but to find any or all subsequences
that are similar between two series. This is used for example to identify that parts of two
series are similar but not necessarily the entire series. Or when comparing a series to itself
it produces subsequences (of arbitrary length) that frequenty reappear in the series.

For example below, we can see that one heartbeat in ECG is a common pattern. Sometimes a sequence
a few heartbeats is similar to another sequence of heartbeats.

::

    lc = local_concurrences(series, None, estimate_settings=0.7)  # second is None to compare to self
    # The parameters tau, delta, delta_factor are estimated based on series
    paths = []
    for match in lc.kbest_matches(k=100, minlen=20, buffer=10):
        paths.append(match.path)
    fig, ax = dtwvis.plot_warpingpaths(series, series, lc.wp, path=-1)
    for path in paths:
        dtwvis.plot_warpingpaths_addpath(ax, path)


.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/localconcurrences.png?v=1
   :alt: Local concurrences

