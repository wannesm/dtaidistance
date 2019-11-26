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


Depending on your system, this might not install the C version. To guarantee installation of the
C extensions (which enable much faster DTW alignment), follow the instructions in the "From Source"
section below.

**Troubleshooting**:

If the C-library is not available after compilation you can try the following two strategies
to identify the problem:

1. Call the `dtw.try_import_c()` function that will print the exception message.
2. Reinstall with `pip install -vvv --upgrade --force-reinstall dtaidistance` and inspect the output.



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
* Compile the C extensions: ``python3 setup.py build_ext --inplace``
* Install into your site-package directory: ``python3 setup.py install``

This requires OpenMP to be available on your system. If this is not the case, use:

::

    $ python3 setup.py --noopenmp build_ext --inplace
