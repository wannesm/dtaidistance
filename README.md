# Time Series Distances

## Usage

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


## Dependencies

- Numpy

Optional:
- Cython


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

