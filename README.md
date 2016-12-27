# Time Series Distances

## Installation

Run `make build` or `python setup.py build_ext --inplace` to be able to use the fast c-based versions of the algorithms.


## Usage

### DTW Distance

Only the distance:

    from dtaidistance import dtw
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    distance = dtw.distance(s1, s2)
    print(distance)

The distance and the full distance matrix:

    from dtaidistance import dtw
    s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
    s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
    distance, matrix = dtw.distances(s1, s2)
    print(distance)
    print(matrix)

The fastest version (30-300 times) uses c directly but requires an array as input:

    from dtaidistance import dtw
    s1 = array.array('d',[0, 0, 1, 2, 1, 0, 1, 0, 0])
    s2 = array.array('d',[0, 1, 2, 0, 0, 0, 0, 0, 0])
    d = dtw.distance_fast(s1, s2)

Or you can use a numpy array:

    from dtaidistance import dtw
    s1 = np.array([0, 0, 1, 2, 1, 0, 1, 0, 0], dtype=np.float64)
    s2 = np.array([0, 1, 2, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    d = dtw.distance_fast(s1, s2)


### DTW Distance Matrix

Use the method `dtw.distance_matrix`. You can set variables to use more or less c code (`use_c` and `use_nogil`) and parallel or serial execution (`parallel`).


## Dependencies

- [Numpy](http://www.numpy.org)

Optional:
- [Cython](http://cython.org)

Development:
- [pytest](http://doc.pytest.org)
- [pytest-benchmark](http://pytest-benchmark.readthedocs.io)


## Contact

- [Wannes Meert](https://people.cs.kuleuven.be/wannes.meert)  
  [Wannes.Meert@cs.kuleuven.be](mailto:Wannes.Meert@cs.kuleuven.be)


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

