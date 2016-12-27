"""
 python3 setup.py build_ext --inplace
"""
from distutils.core import setup, Extension
from distutils.cmd import Command
from Cython.Build import cythonize
import numpy
import platform
import os
import sys
import re


class PyTest(Command):
    user_options = [('pytest-args=', 'a', "Arguments to pass into py.test")]
    pytest_args = []

    def initialize_options(self):
        self.pytest_args = ['--ignore=venv']
        try:
            import pytest_benchmark
            self.pytest_args += ['--benchmark-skip']
        except ImportError:
            pass

    def finalize_options(self):
        pass

    def run(self):
        import pytest

        sys.path.append('.')
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


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

requires = [
    'numpy',
    'cython'
]

tests_require = [
    'pytest'
]

with open('dtaidistance/__init__.py', 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)

if not version:
    raise RuntimeError('Cannot find version information')

setup(
    name='dtaidistance',
    version='0.1.0',
    description='Distance measures for time series',
    long_description=open('README.md').read(),
    author='Wannes Meert',
    author_email='wannes.meert@cs.kuleuven.be',
    url='https://dtai.cs.kuleuven.be',
    requires=requires,
    cmdclass={
        'test': PyTest
    },
    license='Apache 2.0',
    classifiers=(
        'License :: OSI Approved :: Apache Software License'
    ),
    ext_modules=cythonize(extensions),
)
