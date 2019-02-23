#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
 python3 setup.py build_ext --inplace
"""
from setuptools import setup, Command
from setuptools.extension import Extension
from setuptools.command.test import test as TestCommand
from setuptools.command.sdist import sdist as SDistCommand
from setuptools.command.build_ext import build_ext as BuildExtCommand
from setuptools.command.install import install
import platform
import os
import sys
import re

try:
    import numpy
    np_include_dirs = [numpy.get_include()]
except ImportError:
    numpy = None
    np_include_dirs = []

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

use_openmp = False
here = os.path.abspath(os.path.dirname(__file__))


if "--openmp" in sys.argv:
    use_openmp = True


class MyInstallCommand(install):
    user_options = install.user_options + [
        ('openmp', None, 'Use openmp'),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.openmp = None

    def finalize_options(self):
        # print("Use openmp", self.openmp)
        install.finalize_options(self)

    def run(self):
        global use_openmp
        if self.openmp == 1:
            use_openmp = True
            # TODO: too late, Extension has been created
        print(f"Set global use_openmp to {use_openmp}")
        install.run(self)


class MySDistCommand(SDistCommand):
    def run(self):
        PrepReadme.run_pandoc()
        super().run()


class PrepReadme(Command):
    description = "Translate readme from Markdown to ReStructuredText"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        PrepReadme.run_pandoc()

    @staticmethod
    def run_pandoc():
        import subprocess as sp
        print("running pandoc")
        try:
            sp.call(['pandoc', '--from=markdown', '--to=rst', '--output=README', 'README.md'])
        except sp.CalledProcessError as err:
            print("Pandoc failed, Mardown format will be used.")
            print(err)


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
    user_options = BuildExtCommand.user_options + [
        ('openmp', None, 'Use openmp'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.openmp = None
        self.inplace = True

    def finalize_options(self):
        global extra_compile_args
        global extra_link_args
        print("Use openmp", self.openmp)
        if '-fopenmp' not in extra_compile_args:
            extra_compile_args += ['-fopenmp']
        if '-fopenmp' not in extra_link_args:
            extra_link_args += ['-fopenmp']
        super().finalize_options()

    def run(self):
        global use_openmp
        global extra_compile_args
        global extra_link_args
        if self.openmp == 1:
            use_openmp = True
        if '-fopenmp' not in extra_compile_args:
            extra_compile_args += ['-fopenmp']
        if '-fopenmp' not in extra_link_args:
            extra_link_args += ['-fopenmp']
        super().run()


class MyBuildExtInPlaceCommand(MyBuildExtCommand):
    def initialize_options(self):
        super().initialize_options()
        self.inplace = True


extra_compile_args = []
extra_link_args = []
if platform.system() == 'Darwin':
    cppflags = []
    if os.path.exists("/usr/local/opt/llvm/bin/clang"):
        # We have a recent version of LLVM that probably supports openmp to compile parallel C code (installed using
        # `brew install llvm`).
        os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
        os.environ["LDFLAGS"] = "-L/usr/local/opt/llvm/lib"
        cppflags += ["-I/usr/local/opt/llvm/include"]
        # extra_compile_args += ['-fopenmp']
        # extra_link_args += ['-fopenmp']
        try:
            mac_ver = [int(nb) for nb in platform.mac_ver()[0].split(".")]
            if mac_ver[0] == 10 and mac_ver[1] >= 14:
                # From Mojave on, the header files are part of Xcode.app
                cppflags += ['-I/Applications/Xcode.app/Contents/Developer/Platforms/' +
                             'MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include']
        except Exception as exc:
            print("Failed to check version")
            print(exc)
    if len(cppflags) > 0:
        os.environ["CPPFLAGS"] = " ".join(cppflags)

if use_openmp:
    if '-fopenmp' not in extra_compile_args:
        extra_compile_args += ['-fopenmp']
    if '-fopenmp' not in extra_link_args:
        extra_link_args += ['-fopenmp']

if cythonize is not None and numpy is not None:
    print("create ext modules")
    ext_modules = cythonize([
        Extension(
            "dtaidistance.dtw_c", ["dtaidistance/dtw_c.pyx"],
            include_dirs=np_include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)])
elif numpy is None:
    print("Numpy was not found, preparing a pure Python version.")
    ext_modules = []
else:
    print("Cython was not found, preparing a pure Python version.")
    ext_modules = []
    # ext_modules = [
    #     Extension("dtaidistance.dtw_c", ["dtaidistance/dtw_c.c"],
    #               include_dirs=[numpy.get_include()],
    #               extra_compile_args=extra_compile_args,
    #               extra_link_args=extra_link_args)]

install_requires = ['numpy', 'cython']
tests_require = ['pytest', 'matplotlib']

with open('dtaidistance/__init__.py', 'r', encoding='utf-8') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)
if not version:
    raise RuntimeError('Cannot find version information')

readme_path = os.path.join(here, 'README')
if not os.path.exists(readme_path):
    try:
        PrepReadme.run_pandoc()
    except:
        pass
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    with open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='dtaidistance',
    version=version,
    description='Distance measures for time series',
    long_description=long_description,
    author='Wannes Meert',
    author_email='wannes.meert@cs.kuleuven.be',
    url='https://dtai.cs.kuleuven.be',
    project_urls={
        'DTAIDistance documentation': 'http://dtaidistance.readthedocs.io/en/latest/',
        'DTAIDistance source': 'https://github.com/wannesm/dtaidistance'
    },
    packages=["dtaidistance"],
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        'vis': ['matplotlib']
    },
    include_package_data=True,
    package_data={
        '': ['*.pyx', '*.pxd'],
    },
    cmdclass={
        'test': PyTest,
        'readme': PrepReadme,
        'sdist': MySDistCommand,
        'buildinplace': MyBuildExtInPlaceCommand,
        'build_ext': MyBuildExtCommand,
        'install': MyInstallCommand
    },
    license='Apache 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    keywords='dtw',
    ext_modules=ext_modules
)
