OpenMP
------

For DTAIDistance to be able to use parallel computing for the fast C-based code, you need OpenMP to be available on
your system.

OpenMP can be challenging to get to work. On this page, we have collected tips and tricks.


macOS
~~~~~

Via Homebrew, you can use a recent LLVM clang compiler and LLVM's OpenMP library:

::

    brew install llvm libomp

Set environment variables to:

::

    export CC=/usr/local/opt/llvm/bin/clang
    export LDFLAGS="-L/usr/local/opt/llvm/lib"
    export CPPFLAGS="-I/usr/local/opt/llvm/include"

Compile and link using:

::

    clang -Xpreprocessor -fopenmp -lomp myfile.c


Other sources:

- https://iscinumpy.gitlab.io/post/omp-on-high-sierra/

