[![PyPi Version](https://img.shields.io/pypi/v/dtaidistance.svg)](https://pypi.org/project/dtaidistance/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/dtaidistance.svg)](https://anaconda.org/conda-forge/dtaidistance)
[![Documentation Status](https://readthedocs.org/projects/dtaidistance/badge/?version=latest)](https://dtaidistance.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/80764246.svg)](https://zenodo.org/badge/latestdoi/80764246) 

# Time Series Distances

Library for time series distances (e.g. Dynamic Time Warping) used in the
[DTAI Research Group](https://dtai.cs.kuleuven.be). The library offers a pure
Python implementation and a fast implementation in C. The C implementation
has only Cython as a dependency. It is compatible with Numpy and Pandas and
implemented such that unnecessary data copy operations are avoided.

Documentation: http://dtaidistance.readthedocs.io

Example:

    from dtaidistance import dtw
    import numpy as np
    s1 = np.array([0.0, 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2)

Citing this work:

> Wannes Meert, Kilian Hendrickx, Toon Van Craenendonck, Pieter Robberechts, Hendrik Blockeel & Jesse Davis.  
> DTAIDistance (Version v2). Zenodo.  
> http://doi.org/10.5281/zenodo.5901139

**New in v2**:

- Numpy is now an optional dependency, also to compile the C library
  (only Cython is required).
- Small optimizations throughout the C code to improve speed.
- The consistent use of `ssize_t` instead of `int` allows for larger data structures on 64 bit 
  machines and be more compatible with Numpy.
- The parallelization is now implemented directly in C (included if OpenMP is installed).
- The `max_dist` argument turned out to be similar to Silva and Batista's work 
  on PrunedDTW [7]. The toolbox now implements a version that is equal to PrunedDTW
  since it prunes more partial distances. Additionally, a `use_pruning` argument
  is added to automatically set `max_dist` to the Euclidean distance, as suggested
  by Silva and Batista, to speed up the computation (a new method `ub_euclidean` is available).
- Support in the C library for multi-dimensional sequences in the `dtaidistance.dtw_ndim`
  package.
- DTW Barycenter Averaging for clustering (v2.2).
- Subsequence search and local concurrences (v2.3).
- Support for N-dimensional time series (v2.3.7).


## Installation

    $ pip install dtaidistance
    
or

    $ conda install -c conda-forge dtaidistance

The pip installation requires Numpy as a dependency to compile Numpy-compatible
C code (using Cython). However, this dependency is optional and can be removed.

The source code is available at
[github.com/wannesm/dtaidistance](https://github.com/wannesm/dtaidistance).

If you encounter any problems during compilation (e.g. the C-based implementation or OpenMP
is not available), see the 
[documentation](https://dtaidistance.readthedocs.io/en/latest/usage/installation.html)
for more options.

## Usage

### Dynamic Time Warping (DTW) Distance Measure

    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis
    import numpy as np
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
    s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
    path = dtw.warping_path(s1, s2)
    dtwvis.plot_warping(s1, s2, path, filename="warp.png")

![Dynamic Time Warping (DTW) Example](https://people.cs.kuleuven.be/wannes.meert/dtw/dtw_example.png?v=5)


#### DTW Distance Measure Between Two Series

Only the distance measure based on two sequences of numbers:

    from dtaidistance import dtw
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    distance = dtw.distance(s1, s2)
    print(distance)

The fastest version (30-300 times) uses c directly but requires an array as input (with the double type),
and (optionally) also prunes computations by setting `max_dist` to the Euclidean upper bound:

    from dtaidistance import dtw
    import array
    s1 = array.array('d',[0, 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = array.array('d',[0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2, use_pruning=True)

Or you can use a numpy array (with dtype double or float):

    from dtaidistance import dtw
    import numpy as np
    s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double)
    s2 = np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2, use_pruning=True)


Check the `__doc__` for information about the available arguments:

    print(dtw.distance.__doc__)

A number of options are foreseen to early stop some paths the dynamic programming algorithm is exploring or tune
the distance measure computation:

- `window`: Only allow for shifts up to this amount away from the two diagonals.
- `max_dist`: Stop if the returned distance measure will be larger than this value.
- `max_step`: Do not allow steps larger than this value.
- `max_length_diff`: Return infinity if difference in length of two series is larger.
- `penalty`: Penalty to add if compression or expansion is applied (on top of the distance).
- `psi`: Psi relaxation to ignore begin and/or end of sequences (for cylical sequences) [2].
- `use_pruning`: Prune computations based on the Euclidean upper bound.


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
    import random
    import numpy as np
    x = np.arange(0, 20, .5)
    s1 = np.sin(x)
    s2 = np.sin(x - 1)
    random.seed(1)
    for idx in range(len(s2)):
        if random.random() < 0.05:
            s2[idx] += (random.random() - 0.5) / 2
    d, paths = dtw.warping_paths(s1, s2, window=25, psi=2)
    best_path = dtw.best_path(paths)
    dtwvis.plot_warpingpaths(s1, s2, paths, best_path)

![DTW Example](https://people.cs.kuleuven.be/wannes.meert/dtw/warping_paths.png?v=3)

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


### Clustering

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


### Subsequence search

DTAIDistance supports various subsequence search algorithms like
Subsequence Alignment, Subsequence KNN Search and Local Concurrences.
See the [documentation](https://dtaidistance.readthedocs.io/en/latest/usage/subsequence.html)
for more information.

### Motif Discovery

While methods such as `dtw.distance_matrix` and `subsequence.subsequencesearch`
can be used for motif discovery in time series (after windowing), a more
efficient and effective algorithm
based on time warping is available in the
[LoCoMotif package](https://github.com/ML-KULeuven/locomotif).



## Dependencies

- [Python 3](http://www.python.org)

Optional:

- [Cython](http://cython.org)
- [Numpy](http://www.numpy.org)
- [tqdm](https://github.com/tqdm/tqdm)
- [Matplotlib](https://matplotlib.org)
- [SciPy](https://www.scipy.org)
- [PyClustering](https://pyclustering.github.io)

Development:

- [pytest](http://doc.pytest.org)
- [pytest-benchmark](http://pytest-benchmark.readthedocs.io)


## Contact

- https://people.cs.kuleuven.be/wannes.meert


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
7. D. F. Silva and G. E. Batista. 
   [Speeding up all-pairwise dynamic time warping matrix calculation](http://sites.labic.icmc.usp.br/dfs/pdf/SDM_PrunedDTW.pdf),
   In Proceedings of the 2016 SIAM International Conference on Data Mining, pages 837–845. SIAM, 2016.



## License

    DTAI distance code.

    Copyright 2016-2022 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

