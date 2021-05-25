Installation
------------

From PyPI
~~~~~~~~~

This packages is available on PyPI (requires Python 3):

::

    $ pip install dtaidistance


This requires OpenMP to be available on your system. If this is not the case, use:

::

    $ pip install --global-option=--noopenmp dtaidistance

A version compiled without OpenMP (OMP) might raise an exception when parallelization is required.
To avoid this exception, you can force the method to use Python's multiprocessing library
for parallelization by providing the `--use_mp=True` option.

Depending on your system, this might not install the C version. To guarantee installation of the
C extensions (which enable much faster DTW alignment), follow the instructions in the "From Source"
section below.

**Troubleshooting**:

If the C-library is not available after compilation you can try the following two strategies
to identify the problem:

1. Call the ``dtw.try_import_c(verbose=True)`` function that will print the exception message(s).
2. Reinstall with ``pip install -vvv --upgrade --force-reinstall --no-deps --no-binary :all: dtaidistance``
   and inspect the output.



From Github
~~~~~~~~~~~

If you want to install the latest, unreleased version using pip:

::

    $ pip install git+https://github.com/wannesm/dtaidistance.git#egg=dtaidistance

This requires OpenMP to be available on your system. If this is not the case, use:

::

    $ pip install --global-option=--noopenmp git+https://github.com/wannesm/dtaidistance.git#egg=dtaidistance


From source
~~~~~~~~~~~

The library can also be compiled and/or installed directly from source.

* Download the source from https://github.com/wannesm/dtaidistance
* Compile the C extensions:

::

    python3 setup.py build_ext --inplace

* Install into your site-package directory:

::

    python3 setup.py install

This requires OpenMP to be available on your system. If this is not the case, use:

::

    $ python3 setup.py --noopenmp build_ext --inplace

Changelist
~~~~~~~~~~

Version 2.0.0
'''''''''''''

- Numpy is now an optional dependency, also to compile the C library (only Cython is required).
- Small optimizations throughout the C code to improve speed.
- The consistent use of ssize_t instead of int allows for larger data structures on 64 bit machines and be more compatible with Numpy.
- The parallelization is now implemented directly in C (included if OpenMP is installed).
- The max_dist argument turned out to be similar to Silva and Batista's work on PrunedDTW [7]. The toolbox now implements a version that is equal to PrunedDTW since it prunes more partial distances. Additionally, a use_pruning argument is added to automatically set max_dist to the Euclidean distance, as suggested by Silva and Batista, to speed up the computation.
- Support in the C library for multi-dimensional sequences in the dtaidistance.dtw_ndim package.
