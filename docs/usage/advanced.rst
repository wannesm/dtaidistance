Advanced features
-----------------

Use float instead of double
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fast version is based on C, which defaults to the double
datatype. It is possible to compile the C code to use float or
integers if that is required (requires a C compiler and make):

::

    git clone https://github.com/wannesm/dtaidistance.git
    cd dtaidistance/dtaidistance/jinja
    make float
    # make int
    cd ../..
    make build
    pip install .

