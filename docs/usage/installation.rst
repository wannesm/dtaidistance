
Installation
------------

From PyPI
~~~~~~~~~

This packages is available on `PyPI <https://pypi.org/project/dtaidistance/>`_ (requires Python 3):

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

If the C-library is not available after compilation you can try the following steps
to identify the problem:

1. Call the ``dtw.try_import_c(verbose=True)`` function that will print the status of the package.
2. Reinstall with ``pip install -v --upgrade --force-reinstall --no-build-isolation --no-binary dtaidistance dtaidistance``
   and call ``dtw.try_import_c(verbose=True)`` again.
   The ``--no-build-isolation`` is present to use your already installed versions of Cython and
   Numpy instead of downloading recent versions in an isolation build environment
   (`PEP 517 <https://peps.python.org/pep-0517/>`_). When you are using an older version
   of Numpy, the pre-compiled package might trigger binary incompatibility errors.

**Troubleshootimg (OMP)**:

If the OMP library is not detected during compilation, parallel execution in c is not available.
If OMP is installed but not found, there is probably an issue with the options given to the
compiler. A few variations are available to try alternative options:

::

    # To include -lgomp (when using GOMP instead of OMP)
    $ pip install --global-option=--forcegnugcc dtaidistance
    # To include -lomp:
    $ pip install --global-option=--forcellvm dtaidistance
    # To remove the -Xpreprocessor option (can be combined with the above):
    $ pip install --global-option=--noxpreprocessor dtaidistance

If problems persist, consider using the `Anaconda.org <https://anaconda.org>`_ Python environment (see next section)
for which precompiled versions are available.


From Conda / Anaconda
~~~~~~~~~~~~~~~~~~~~~

This package is available on `anaconda.org <https://anaconda.org/conda-forge/dtaidistance>`_
(incuding precompiled binary versions for Linux, Macos, and Windows):

::

    $ conda install -c conda-forge dtaidistance


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

In case OpenMP is available but the compiler is unable to detect the library, a few
options are available to change the compiler arguments:

- ``--forcegnugcc``: Include the -lgomp argument
- ``--forcellvm``: Include the  -lomp argument
- ``--noxpreprocessor``: Remove the -Xpreprocessor argument
- ``python3 setup.py -h``: To see al options

**Without Numpy**

Most of the dtaidistance package works just fine without Numpy. It is required at
installation because most deployments require Numpy support
(to feed Numpy arrays as input) and therefore the package needs to be
compiled with Numpy support.

If you want to remove the Numpy dependency, remove it from ``pyproject.toml`` file.


From C
~~~~~~

A number of algorithms (DTW, Barycenter averaging) are implemented in C.
They can be called directly from C source code as they do not rely on
Python. All files can be found in ``dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/``.
An example Makefile and XCode project are available. Example usage can be seen
in the ``dd_benchmark.c``, ``dd_tests_dtw.c``, and ``dd_tests_matrix.c`` files.

For example:

::

    $ gcc -c -o DTAIDistanceC/dd_benchmark.o DTAIDistanceC/dd_benchmark.c -Wall -g -Xpreprocessor -fopenmp
    $ gcc -c -o DTAIDistanceC/dd_dtw_openmp.o DTAIDistanceC/dd_dtw_openmp.c -Wall -g -Xpreprocessor -fopenmp
    $ gcc -c -o DTAIDistanceC/dd_ed.o DTAIDistanceC/dd_ed.c -Wall -g -Xpreprocessor -fopenmp
    $ gcc -o dd_benchmark DTAIDistanceC/dd_benchmark.o DTAIDistanceC/dd_dtw.o DTAIDistanceC/dd_dtw_openmp.o DTAIDistanceC/dd_ed.o -Wall -g -Xpreprocessor -fopenmp -lomp
    $ ./dd_benchmark
    Benchmarking ...
    OpenMP is supported
    Creating result array of size 17997000
    Execution time = 7.000000

