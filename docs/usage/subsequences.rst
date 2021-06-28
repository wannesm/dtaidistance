Subsequences
------------

Subsequence search is to match the best occurance of a short time serise in a longer series.

Using DTW
~~~~~~~~~

Given a series:

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/subsequence_series.png?v=1
   :alt: Subsequence series

And a query:

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/subsequence_query.png?v=1
   :alt: Subsequence query

We can find the best occurence(s) as follows:

::

    from dtaidistance.subsequence.dtw import subsequence_search
    from dtaidistance import dtw_visualisation as dtwvis

    sa = subsequence_search(query, series)
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

