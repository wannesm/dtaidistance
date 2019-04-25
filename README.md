# Time Series Distances

Library for time series distances (e.g. Dynamic Time Warping) used in the [DTAI Research Group](https://dtai.cs.kuleuven.be).
The library offers a pure Python implementation and a faster implementation in C.

Documentation: http://dtaidistance.readthedocs.io

Citing this work: [![DOI](https://zenodo.org/badge/80764246.svg)](https://zenodo.org/badge/latestdoi/80764246)


## Installation

This packages is available on PyPI (requires Python 3):

    $ pip install dtaidistance

In case the C based version is not available, see the documentation for alternative installation options.
In case OpenMP is not available on your system add the `--noopenmp` global option.

The source code is available at [github.com/wannesm/dtaidistance](https://github.com/wannesm/dtaidistance).


## Usage

### Dynamic Time Warping (DTW) Distance Measure

    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis
    import numpy as np
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
    s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
    path = dtw.warping_path(s1, s2)
    dtwvis.plot_warping(s1, s2, path, filename="warp.png")

![Dynamic Time Warping (DTW) Example](https://people.cs.kuleuven.be/wannes.meert/dtw/dtw_example.png?v=4)


#### DTW Distance Measure Between Two Series

Only the distance measure based on two sequences of numbers:

    from dtaidistance import dtw
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    distance = dtw.distance(s1, s2)
    print(distance)

The fastest version (30-300 times) uses c directly but requires an array as input (with the double type):

    from dtaidistance import dtw
    import array
    s1 = array.array('d',[0, 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = array.array('d',[0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2)

Or you can use a numpy array (with dtype double or float):

    from dtaidistance import dtw
    import numpy as np
    s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double)
    s2 = np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2)


Check the `__doc__` for information about the available arguments:

    print(dtw.distance.__doc__)

A number of options are foreseen to early stop some paths the dynamic programming algorithm is exploring or tune
the distance measure computation:

- `window`: Only allow for shifts up to this amount away from the two diagonals.
- `max_dist`: Stop if the returned distance measure will be larger than this value.
- `max_step`: Do not allow steps larger than this value.
- `max_length_diff`: Return infinity if difference in length of two series is larger.
- `penalty`: Penalty to add if compression or expansion is applied (on top of the distance).
- `psi`: Psi relaxation to ignore begin and/or end of sequences (for cylical sequencies) [2].


#### DTW Distance Measure all warping paths

If, next to the distance, you also want the full matrix to see all possible warping paths:

    from dtaidistance import dtw
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    distance, paths = dtw.warping_paths(s1, s2)
    print(distance)
    print(paths)

The matrix with all warping paths can be visualised as follows:

    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis
    import numpy as np
    x = np.arange(0, 20, .5)
    s1 = np.sin(x)
    s2 = np.sin(x - 1)
    d, paths = dtw.warping_paths(s1, s2, window=25, psi=2)
    best_path = dtw.best_path(paths)
    dtwvis.plot_warpingpaths(s1, s2, paths, best_path)

![DTW Example](https://people.cs.kuleuven.be/wannes.meert/dtw/warping_paths.png?v=2)

Notice the `psi` parameter that relaxes the matching at the beginning and end.
In this example this results in a perfect match even though the sine waves are slightly shifted.


#### DTW Distance Measures Between Set of Series

To compute the DTW distance measures between all sequences in a list of sequences, use the method `dtw.distance_matrix`.
You can set variables to use more or less c code (`use_c` and `use_nogil`) and parallel or serial execution
(`parallel`).

The `distance_matrix` method expects a list of lists/arrays:

    from dtaidistance import dtw
    import numpy as np
    series = [
        np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double),
        np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0.0, 0, 1, 2, 1, 0, 0, 0])]
    ds = dtw.distance_matrix_fast(series)

or a matrix (in case all series have the same length):

    from dtaidistance import dtw
    import numpy as np
    series = np.matrix([
        [0.0, 0, 1, 2, 1, 0, 1, 0, 0],
        [0.0, 1, 2, 0, 0, 0, 0, 0, 0],
        [0.0, 0, 1, 2, 1, 0, 0, 0, 0]])
    ds = dtw.distance_matrix_fast(series)


#### DTW Distance Measures Between Set of Series, limited to block

You can instruct the computation to only fill part of the distance measures matrix.
For example to distribute the computations over multiple nodes, or to only 
compare source series to target series.

    from dtaidistance import dtw
    import numpy as np
    series = np.matrix([
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]])
    ds = dtw.distance_matrix_fast(series, block=((1, 4), (3, 5)))

The output in this case will be:

    #  0     1    2    3       4       5
    [[ inf   inf  inf     inf     inf  inf]    # 0
     [ inf   inf  inf  1.4142  0.0000  inf]    # 1
     [ inf   inf  inf  2.2360  1.7320  inf]    # 2
     [ inf   inf  inf     inf  1.4142  inf]    # 3
     [ inf   inf  inf     inf     inf  inf]    # 4
     [ inf   inf  inf     inf     inf  inf]]   # 5


## Clustering

A distance matrix can be used for time series clustering. You can use existing methods such as
`scipy.cluster.hierarchy.linkage` or one of two included clustering methods (the latter is a
wrapper for the SciPy linkage method).

    from dtaidistance import clustering
    # Custom Hierarchical clustering
    model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
    cluster_idx = model1.fit(series)
    # Augment Hierarchical object to keep track of the full tree
    model2 = clustering.HierarchicalTree(model1)
    cluster_idx = model2.fit(series)
    # SciPy linkage clustering
    model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
    cluster_idx = model3.fit(series)


For models that keep track of the full clustering tree (`HierarchicalTree` or `LinkageTree`), the
tree can be visualised:

    model.plot("myplot.png")

![Dynamic Time Warping (DTW) hierarchical clusteringt](https://people.cs.kuleuven.be/wannes.meert/dtw/hierarchy.png?v=2)


## Dependencies

- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)

Optional:

- [Cython](http://cython.org)
- [tqdm](https://github.com/tqdm/tqdm)
- [matplotlib](https://matplotlib.org)

Development:

- [pytest](http://doc.pytest.org)
- [pytest-benchmark](http://pytest-benchmark.readthedocs.io)


## Contact

- [Wannes Meert](https://people.cs.kuleuven.be/wannes.meert)  
  <[Wannes.Meert@cs.kuleuven.be](mailto:Wannes.Meert@cs.kuleuven.be)>


## References

1. T. K. Vintsyuk,
   Speech discrimination by dynamic programming.
   Kibernetika, 4:81–88, 1968.
2. H. Sakoe and S. Chiba,
   Dynamic programming algorithm optimization for spoken word recognition.
   IEEE Transactions on Acoustics, Speech and Signal Processing, 26(1):43–49, 1978.
3. C. S. Myers and L. R. Rabiner,
   A comparative study of several dynamic time-warping algorithms for connected-word recognition.
   The Bell System Technical Journal, 60(7):1389–1409, Sept 1981.
4. Mueen, A and Keogh, E, 
   [Extracting Optimal Performance from Dynamic Time Warping](http://www.cs.unm.edu/~mueen/DTW.pdf),
   Tutorial, KDD 2016
5. D. F. Silva, G. E. A. P. A. Batista, and E. Keogh.
   [On the effect of endpoints on dynamic time warping](http://www-bcf.usc.edu/~liu32/milets16/paper/MiLeTS_2016_paper_7.pdf),
   In SIGKDD Workshop on Mining and Learning from Time Series, II. Association for Computing Machinery-ACM, 2016.
6. C. Yanping, K. Eamonn, H. Bing, B. Nurjahan, B. Anthony, M. Abdullah and B. Gustavo.
   [The UCR Time Series Classification Archive](www.cs.ucr.edu/~eamonn/time_series_data/), 2015.


## License

    DTAI distance code.

    Copyright 2016-2019 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

