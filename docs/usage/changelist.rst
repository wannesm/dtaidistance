Changelist
----------

Version 2.3
~~~~~~~~~~~

- Subsequence search and local concurrences
- Parallellization improvements in C-code for >8 threads
  (thanks to Erlend Kvinge Jørgensen)

Version 2.2
~~~~~~~~~~~

- DTW Barycenter Averaging
- K-means DBA clustering

Version 2.1
~~~~~~~~~~~

- Various improvements in the C code
- K-medoids clustering

Version 2.0
~~~~~~~~~~~

- Numpy is now an optional dependency, also to compile the C library (only Cython is required).
- Small optimizations throughout the C code to improve speed.
- The consistent use of ssize_t instead of int allows for larger data structures on 64 bit machines and be more compatible with Numpy.
- The parallelization is now implemented directly in C (included if OpenMP is installed).
- The max_dist argument turned out to be similar to Silva and Batista's work on PrunedDTW [7]. The toolbox now implements a version that is equal to PrunedDTW since it prunes more partial distances. Additionally, a use_pruning argument is added to automatically set max_dist to the Euclidean distance, as suggested by Silva and Batista, to speed up the computation.
- Support in the C library for multi-dimensional sequences in the dtaidistance.dtw_ndim package.
