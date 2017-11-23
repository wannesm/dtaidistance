# Time Series Distances

Experimental library for time series distances used in the [DTAI Research Group](https://dtai.cs.kuleuven.be).

## Installation

The library can be used as a pure Python implementation. If you need a faster version of the algorithms you can make use of the included C algorithms. You might need to run `make build` or `python setup.py build_ext --inplace` to compile the included library first.


## Usage

### Dynamic Time Warping (DTW) Distance

    from dtaidistance import dtw
    s1 = np.array([0., 0, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0, 0])
    s2 = np.array([0., 1, 2, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0])
    dtw.plot_warping(s1, s2)

![DTW Example](https://people.cs.kuleuven.be/wannes.meert/dtw/dtw_example.png?v=2)


#### DTW Distance Between Two Series

Only the distance based on two sequences of numbers:

    from dtaidistance import dtw
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    distance = dtw.distance(s1, s2)
    print(distance)

Check the `__doc__` for information about the available arguments:

    print(dtw.distance.__doc__)

If, next to the distance, you also want the full distance matrix:

    from dtaidistance import dtw
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    distance, matrix = dtw.distances(s1, s2)
    print(distance)
    print(matrix)

The fastest version (30-300 times) uses c directly but requires an array as input (with the double type):

    from dtaidistance import dtw
    s1 = array.array('d',[0, 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = array.array('d',[0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2)

Or you can use a numpy array (with dtype double or float):

    from dtaidistance import dtw
    s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double)
    s2 = np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2)


#### DTW Distances Between Set of Series

To compute the DTW distances between all sequences in a list of sequences, use the method `dtw.distance_matrix`.
You can set variables to use more or less c code (`use_c` and `use_nogil`) and parallel or serial execution
(`parallel`).

The `distance_matrix` method expects a list of lists/arrays or a matrix (in case all series have the same length).

    from dtaidistance import dtw
    series = [
        np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.double),
        np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0.0, 0, 1, 2, 1, 0, 0, 0])]
    ds = dtw.distance_matrix_fast(s)

    from dtaidistance import dtw
    series = np.matrix([
        [0.0, 0, 1, 2, 1, 0, 1, 0, 0],
        [0.0, 1, 2, 0, 0, 0, 0, 0, 0],
        [0.0, 0, 1, 2, 1, 0, 0, 0, 0]])
    ds = dtw.distance_matrix_fast(s)


#### Distribute DTW Distance computations over blocks

You can instruct the computation to only fill part of the distances matrix.
For example to distribute the computations over multiple nodes.

    from dtaidistance import dtw
    series = np.matrix([
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1],
         [0., 0, 1, 2, 1, 0, 1, 0, 0],
         [0., 1, 2, 0, 0, 0, 0, 0, 0],
         [1., 2, 0, 0, 0, 0, 0, 1, 1]])
    ds = dtw.distance_matrix_fast(s, block=((1, 4), (3, 5)))

The output will be:

    #  0     1    2    3           4           5
    [[ inf   inf  inf     inf     inf  inf]    # 0
     [ inf   inf  inf  1.4142  0.0000  inf]    # 1
     [ inf   inf  inf  2.2360  1.7320  inf]    # 2
     [ inf   inf  inf     inf  1.4142  inf]    # 3
     [ inf   inf  inf     inf     inf  inf]    # 4
     [ inf   inf  inf     inf     inf  inf]]   # 5


## Dependencies

- [Numpy](http://www.numpy.org)

Optional:
- [Cython](http://cython.org)
- [tqdm](https://github.com/tqdm/tqdm)

Development:
- [pytest](http://doc.pytest.org)
- [pytest-benchmark](http://pytest-benchmark.readthedocs.io)


## Contact

- [Wannes Meert](https://people.cs.kuleuven.be/wannes.meert)  
  [Wannes.Meert@cs.kuleuven.be](mailto:Wannes.Meert@cs.kuleuven.be)


## References

1. Mueen, A and Keogh, E, 
   [Extracting Optimal Performance from Dynamic Time Warping](http://www.cs.unm.edu/~mueen/DTW.pdf),
   Tutorial, KDD 2016


## License

    DTAI distance code.

    Copyright 2016 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

