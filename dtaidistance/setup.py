"""
 python3 setup.py build_ext --inplace
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

print(numpy.get_include())

extensions = [
    Extension("dtw_c", ["dtw_c.pyx"],
        include_dirs=[numpy.get_include()]),
]

setup(
    ext_modules=cythonize(extensions),
)

