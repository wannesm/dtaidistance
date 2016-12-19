"""
 python3 setup.py build_ext --inplace
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import platform
import os
import sys

if sys.argv[-1] == "build":
    os.system("python3 setup.py build_ext --inplace")
    sys.exit()

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

required = [
    'numpy',
    'cython'
]

setup(
    name='dtaidistance',
    version='0.1.0',
    description='Distance measures for time series',
    long_descrption=open('README.md').read(),
    author='Wannes Meert',
    author_email='wannes.meert@cs.kuleuven.be',
    url='https://dtai.cs.kuleuven.be',
    my_modules=['dtaidistance'],
    install_requires=required,
    license='APL',
    classifiers=(),
    ext_modules=cythonize(extensions),
)
