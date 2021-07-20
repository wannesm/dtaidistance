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
