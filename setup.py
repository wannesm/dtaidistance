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
from setuptools import Distribution
import platform
import os
import sys
import re
import subprocess as sp

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

here = os.path.abspath(os.path.dirname(__file__))

c_args = {
    # Xpreprocessor is required for the built-in CLANG on macos, but other
    # installations of LLVM don't seem to be bothered by it (although it's
    # not required.
    'unix': ['-Xpreprocessor', '-fopenmp'],
    'msvc': ['/openmp', '/Ox', '/fp:fast', '/favor:INTEL64', '/Og'],
    'mingw32': ['-fopenmp', '-O3', '-ffast-math', '-march=native']
}
l_args = {
    'unix': ['-fopenmp'],
    'msvc': [],
    'mingw32': ['-fopenmp']
}


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
        import pytest
        sys.path.append('.')
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


class MyDistribution(Distribution):
    global_options = Distribution.global_options + [
        ('noopenmp', None, 'Disable compilation with openmp')
    ]

    def __init__(self, attrs=None):
        self.noopenmp = 0
        super().__init__(attrs)


class MyInstallCommand(install):
    pass

    # def initialize_options(self):
    #     install.initialize_options(self)

    # def finalize_options(self):
    #     install.finalize_options(self)

    # def run(self):
    #     install.run(self)


def set_custom_envvars_for_homebrew():
    """Update environment variables automatically for Homebrew if CC is not set"""
    if platform.system() == 'Darwin' and "CC" not in os.environ:
        print("Set custom environment variables for Homebrew Clang because CC is not set")
        cppflags = []
        if "CPPFLAGS" in os.environ:
            cppflags.append(os.environ["CPPFLAGS"])
        cflags = []
        if "CFLAGS" in os.environ:
            cflags.append(os.environ["CFLAGS"])
        ldflags = []
        if "LDFLAGS" in os.environ:
            ldflags.append(os.environ["LDFLAGS"])
        if os.path.exists("/usr/local/opt/llvm/bin/clang"):
            # We have a recent version of LLVM that probably supports openmp to compile parallel C code (installed using
            # `brew install llvm`).
            os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
            print("CC={}".format(os.environ["CC"]))
            ldflags += ["-L/usr/local/opt/llvm/lib"]
            cppflags += ["-I/usr/local/opt/llvm/include"]
            cflags += ["-I/usr/local/opt/llvm/include"]
            try:
                mac_ver = [int(nb) for nb in platform.mac_ver()[0].split(".")]
                if mac_ver[0] == 10 and mac_ver[1] >= 14:
                    # From Mojave on, the header files are part of Xcode.app
                    incpath = '-I/Applications/Xcode.app/Contents/Developer/Platforms/' + \
                              'MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include'
                    cppflags += [incpath]
                    cflags += [incpath]
            except Exception as exc:
                print("Failed to check version")
                print(exc)
        else:
            # The default clang in XCode is compatible with OpenMP when using -Xpreprocessor
            pass

        if len(cppflags) > 0:
            os.environ["CPPFLAGS"] = " ".join(cppflags)
            print("CPPFLAGS={}".format(os.environ["CPPFLAGS"]))
        if len(cflags) > 0:
            os.environ["CFLAGS"] = " ".join(cflags)
            print("CFLAGS={}".format(os.environ["CFLAGS"]))
        if len(ldflags) > 0:
            os.environ["LDFLAGS"] = " ".join(ldflags)
            print("LDFLAGS={}".format(os.environ["LDFLAGS"]))
    else:
        print("Using the following environment variables:")
        print("CC={}".format(os.environ.get("CC", "")))
        print("CPPFLAGS={}".format(os.environ.get("CPPFLAGS", "")))
        print("CFLAGS={}".format(os.environ.get("CPLAGS", "")))
        print("LDFLAGS={}".format(os.environ.get("LDFLAGS", "")))


class MyBuildExtCommand(BuildExtCommand):

    def build_extensions(self):
        c = self.compiler.compiler_type
        print("Compiler type: {}".format(c))
        print("--noopenmp: {}".format(self.distribution.noopenmp))
        if self.distribution.noopenmp == 0:
            try:
                check_result = check_openmp(self.compiler.compiler[0])
            except Exception as exc:
                print("WARNING: Cannot check for OpenMP, assuming to be available")
                print(exc)
                check_result = True  # Assume to be present by default
            if not check_result:
                print("WARNING: OpenMP is not available, disabling OpenMP (no parallel computing in C)")
                self.distribution.noopenmp = 1
        if c in c_args:
            if self.distribution.noopenmp == 1:
                args = [arg for arg in c_args[c] if "openmp" not in arg]
            else:
                args = c_args[c]
            for e in self.extensions:
                e.extra_compile_args = args
        else:
            print("Unknown compiler type: {}".format(c))
        if c in l_args:
            if self.distribution.noopenmp == 1:
                args = [arg for arg in l_args[c] if "openmp" not in arg]
            else:
                args = l_args[c]
            for e in self.extensions:
                e.extra_link_args = args
        if numpy is None:
            self.extensions = [arg for arg in self.extensions if "numpy" not in str(arg)]
        print(f'All extensions:')
        print(self.extensions)
        BuildExtCommand.build_extensions(self)

    def initialize_options(self):
        set_custom_envvars_for_homebrew()
        super().initialize_options()

    # def finalize_options(self):
    #     super().finalize_options()

    # def run(self):
    #     super().run()


class MyBuildExtInPlaceCommand(MyBuildExtCommand):
    def initialize_options(self):
        super().initialize_options()
        self.inplace = True


def check_openmp(cc_bin):
    """Check if OpenMP is available"""
    print("Checking for OpenMP availability")
    cc_binname = os.path.basename(cc_bin)
    args = None
    kwargs = None
    if "clang" in cc_binname or "cc" in cc_binname:
        args = [[str(cc_bin), "-dM", "-E", "-fopenmp", "-"]]
        kwargs = {"stdout": sp.PIPE, "input": '', "encoding": 'ascii'}
        print(" ".join(args[0]) + " ".join(str(k) + "=" + str(v) for k, v in kwargs.items()))
    if args is not None:
        try:
            p = sp.run(*args, **kwargs)
            defs = p.stdout.splitlines()
            for curdef in defs:
                if "_OPENMP" in curdef:
                    print("... found OpenMP")
                    return True
        except Exception:
            print("... no OpenMP")
            return False
    else:
        print("... do not know how to check for OpenMP (unknown CC), assume to be available")
        return True
    return False


# Set up extension
extensions = []
if cythonize is not None:
    os.environ["CPPFLAGS"] = "-Idtaidistance/lib/DTAIDistanceC/DTAIDistanceC" + " " + os.environ["CPPFLAGS"]
    extensions.append(
        Extension(
            "dtaidistance.dtw_cc",
            ["dtaidistance/dtw_cc.pyx",
             "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dtw.c"],
            include_dirs=[],
            extra_compile_args=[],
            extra_link_args=[]))
    extensions.append(
        Extension(
            "dtaidistance.ed_cc",
            ["dtaidistance/ed_cc.pyx",
             "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/ed.c"],
            include_dirs=[],
            extra_compile_args=[],
            extra_link_args=[]))
    extensions.append(
        Extension(
            "dtaidistance.dtw_cc_omp",
            ["dtaidistance/dtw_cc_omp.pyx",
             "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dtw_openmp.c"],
            include_dirs=[],
            extra_compile_args=[],
            extra_link_args=[]))

    if numpy is not None:
        extensions.append(
            Extension(
                "dtaidistance.dtw_cc_numpy", ["dtaidistance/util_numpy_cc.pyx"],
                include_dirs=np_include_dirs,
                extra_compile_args=[],
                extra_link_args=[]))
        # extensions.append(
        #     Extension(
        #         "dtaidistance.dtw_c", ["dtaidistance/dtw_c.pyx"],
        #         include_dirs=np_include_dirs,
        #         extra_compile_args=[],
        #         extra_link_args=[]))
    else:
        print("WARNING: Numpy was not found, preparing a version without Numpy support.")

    ext_modules = cythonize(extensions)
else:
    print("WARNING: Cython was not found, preparing a pure Python version.")
    ext_modules = []

install_requires = ['cython']
tests_require = ['pytest']

# Check version number
with open('dtaidistance/__init__.py', 'r', encoding='utf-8') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)
if not version:
    raise RuntimeError('Cannot find version information')

# Set up readme file
readme_path = os.path.join(here, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = ""
long_description_content_type = "text/markdown"

# Create setup
setup(
    name='dtaidistance',
    version=version,
    description='Distance measures for time series',
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    author='Wannes Meert',
    author_email='wannes.meert@cs.kuleuven.be',
    url='https://dtai.cs.kuleuven.be',
    project_urls={
        'DTAIDistance documentation': 'http://dtaidistance.readthedocs.io/en/latest/',
        'DTAIDistance source': 'https://github.com/wannesm/dtaidistance'
    },
    packages=["dtaidistance"],
    python_requires='>=3.5',
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={
        'vis': ['matplotlib'],
        'numpy': ['numpy']
    },
    include_package_data=True,
    package_data={
        '': ['*.pyx', '*.pxd', '*.c', '*.h'],
    },
    distclass=MyDistribution,
    cmdclass={
        'test': PyTest,
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
