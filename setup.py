#!/usr/bin/env python3
"""
 python3 setup.py build_ext --inplace
"""
from setuptools import setup, Command
from setuptools.extension import Extension
from setuptools.command.test import test as TestCommand
from setuptools.command.sdist import sdist as SDistCommand
from setuptools.command.build_ext import build_ext as BuildExtCommand
import numpy
import platform
import os
import sys
import re

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

here = os.path.abspath(os.path.dirname(__file__))


class MySDistCommand(SDistCommand):
    def run(self):
        # How do we call another command?
        import subprocess as sp
        print("prepare README file")
        sp.call(['pandoc', '--from=markdown', '--to=rst', '--output=README', 'README.md'])
        super().run()


class PrepReadme(Command):
    description = "Translate readme from Markdown to ReStructuredText"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess as sp
        sp.call(['pandoc', '--from=markdown', '--to=rst', '--output=README', 'README.md'])


class PyTest(TestCommand):
    description = "Run tests"
    user_options = [('pytest-args=', 'a', "Arguments to pass into py.test")]
    pytest_args = []
    test_args = []

    def initialize_options(self):
        self.pytest_args = ['--ignore=venv']
        try:
            import pytest_benchmark
            self.pytest_args += ['--benchmark-skip']
        except ImportError:
            print("No benchmark library, ignore benchmarks")
            self.pytest_args += ['--ignore', 'tests/test_benchmark.py']

    def finalize_options(self):
        pass

    def run_tests(self):
        # import shlex
        import pytest
        sys.path.append('.')
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


class MyBuildExtCommand(BuildExtCommand):
    def initialize_options(self):
        super().initialize_options()
        self.inplace = True


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

if cythonize:
    ext_modules = cythonize([
        Extension(
            "dtaidistance.dtw_c", ["dtaidistance/dtw_c.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)])
else:
    ext_modules = [
        Extension("dtaidistance.dtw_c", ["dtaidistance/dtw_c.c"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args)]

install_requires = ['numpy']
tests_require = ['pytest', 'cython']

with open('dtaidistance/__init__.py', 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)
if not version:
    raise RuntimeError('Cannot find version information')

with open(os.path.join(here, 'README'), 'r') as f:
    long_description = f.read()

setup(
    name='dtaidistance',
    version=version,
    description='Distance measures for time series',
    long_description=long_description,
    author='Wannes Meert',
    author_email='wannes.meert@cs.kuleuven.be',
    url='https://dtai.cs.kuleuven.be',
    packages=["dtaidistance"],
    install_requires=install_requires,
    tests_require=tests_require,
    cmdclass={
        'test': PyTest,
        'readme': PrepReadme,
        'sdist': MySDistCommand,
        'buildinplace': MyBuildExtCommand
    },
    license='Apache 2.0',
    classifiers=(
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        # 'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ),
    keywords='dtw',
    ext_modules=ext_modules
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
