Installation
------------

From PyPI
~~~~~~~~~

This packages is available on PyPI:

::

    $ pip install dtaidistance

In case the C based version is not available, you might need to run
``make build`` or ``python3 setup.py build_ext --inplace`` to compile the
included library first.


From Github
~~~~~~~~~~~

If you want to install the latest, unreleased version using pip:

::

    $ pip install git+https://github.com/wannesm/dtaidistance.git#egg=dtaidistance


From source
~~~~~~~~~~~

The library can also be compiled and/or installed directly from source.

* Download the source from https://github.com/wannesm/dtaidistance
* To compile and install in your site-package directory: ``python3 setup.py install``
* To compile locally: ``python3 setup.py build_ext --inplace``
