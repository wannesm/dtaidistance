Explain DTW (DSW)
~~~~~~~~~~~~~~~~~

Comparing time series is essential in various tasks such as clustering and
classification. While elastic distance measures that allow warping, such as
Dynamic Time Warping (DTW), provide a robust quantitative comparison, a
qualitative comparison is difficult. Traditional visualizations focus on
point-to-point alignment and do not convey the broader structural relationships
at the level of subsequences. This makes it difficult to understand how and
where one time series shifts, speeds up or slows down with respect to another.
Dynamic Subsequence Warping is a method that simplifies the warping path to
highlight, quantify and visualize key transformations (shift, compression,
difference in amplitude). This representation of how subsequences match between
time series.

DSW is based on the following paper:

    Lin, S., Meert, W. Robberechts, P., Blockeel H.,
    "Warping and Matching Subsequences Between Time Series"
    arXiv:2506.15452v1 [cs.LG] 2025 
    (`https://arxiv.org/abs/2506.15452 <https://arxiv.org/abs/2506.15452>`__)
    

To plot the dynamic subsequence warping (DSW) explanation:

::

    ep = ExplainPair(ya, yb, delta_rel=2, delta_abs=0.5, split_strategy=SplitStrategy.DERIV_DIST)
    ep.plot_warping(filename="/path/to/figure.png")


.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/explain_example1_dsw.png?v=1
   :alt: DSW explain example

This visualization is easier to interpret than a point-to-point connected
visualization:

::

    dist, paths = warping_paths(ya, yb)
    path = best_path(paths)
    plot_warping(ya, yb, path, filename="/path/to/figure1.png")
    plot_warping(ya, yb, path, filename="/path/to/figure2.png",
                 start_on_curve=False, color_misalignment=True)

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/explain_example1_dtw.png?v=1
   :alt: DTW explain example


The advantage is even more clear if there is overfitting on the noise present
in the time series. In the following example one can see how the
point-to-point matching has extreme compression and expansion that is not
representive of the actual differences.

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/explain_example2_dsw.png?v=1
   :alt: DSW explain example

.. figure:: https://people.cs.kuleuven.be/wannes.meert/dtw/explain_example2_dtw.png?v=1
   :alt: DTW explain example


