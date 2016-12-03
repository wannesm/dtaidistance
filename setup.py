"""
 python3 setup.py build_ext --inplace
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import platform
import os

print(numpy.get_include())

extra_compile_args = []
extra_link_args = []
if platform.system() == 'Darwin':
    if os.path.exists("/usr/local/opt/llvm/bin/clang"):
        # We have a recent version of LLVM that probably supports openmp to compile parallel C-code (installed using
        # `brew install llvm`).
        os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
        os.environ["LDFLAGS"] = "-L/usr/local/opt/llvm/lib"
        os.environ["CPPFLAGS"] = "-I/usr/local/opt/llvm/include"
        extra_compile_args += ['-fopenmp']
        extra_link_args += ['-fopenmp']

extensions = [
    Extension(
        "dtaidistance.dtw_c", ["dtaidistance/dtw_c.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=cythonize(extensions),
)

