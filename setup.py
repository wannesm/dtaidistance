#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
 python3 setup.py build_ext --inplace
"""
from setuptools import setup, Command, find_packages
from setuptools.extension import Extension
from setuptools.command.test import test as TestCommand
from setuptools.command.sdist import sdist as SDistCommand
from setuptools.command.build_ext import build_ext as BuildExtCommand
from setuptools.command.install import install
from setuptools import Distribution
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
import platform
import os
import sys
import re
import subprocess as sp
from pathlib import Path

try:
    import numpy
except ImportError:
    numpy = None

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

here = Path(__file__).parent
dtaidistancec_path = Path('dtaidistance') / 'lib' / 'DTAIDistanceC' / 'DTAIDistanceC'

c_args = {
    # Xpreprocessor is required for the built-in CLANG on macos, but other
    # installations of LLVM don't seem to be bothered by it (although it's
    # not required.
    # GCC should also not be bothered by it but appears to be on some systems.
    'unix': ['-Xpreprocessor', '-fopenmp',
             '-I'+str(dtaidistancec_path)],
    'msvc': ['/openmp', '/Ox', '/fp:fast', '/favor:INTEL64', '/Og',
             '/I'+str(dtaidistancec_path)],
    'mingw32': ['-fopenmp', '-O3', '-ffast-math', '-march=native', '-DMS_WIN64',
                '-I'+str(dtaidistancec_path)],
    'llvm': ['-Xpreprocessor', '-fopenmp',  # custom key for Homebrew llvm
                  '-I'+str(dtaidistancec_path)],
    'gnugcc': ['-Xpreprocessor', '-fopenmp',  # custom key for GNU GCC
             '-I'+str(dtaidistancec_path)]
}
l_args = {
    'unix': ['-Xpreprocessor', '-fopenmp'],  # '-lgomp' / '-lomp'
    'msvc': [],
    'mingw32': ['-fopenmp'],
    'llvm': ['-Xpreprocessor', '-fopenmp', '-lomp'], # custom key for Homebrew llvm
    'gnugcc': ['-Xpreprocessor', '-fopenmp', '-lgomp'] # custom key for GNU GCC
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
            self.pytest_args += ['--ignore', str(Path('tests') / 'test_benchmark.py')]

    def finalize_options(self):
        pass

    def run_tests(self):
        import pytest
        sys.path.append('.')
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


class MyDistribution(Distribution):
    global_options = Distribution.global_options + [
        ('noopenmp', None, 'No compiler/linker flags for OpenMP (OpenMP disabled)'),
        ('forceopenmp', None, 'Force compiler/linker flags to use OpenMP'),
        ('noxpreprocessor', None, 'Assume OpenMP is built-in (remove -Xpreprocessor argument)'),
        ('forcellvm', None, 'Force compile/linker flags for LLVM (include -lomp argument)'),
        ('forcegnugcc', None, 'Force compile/linker flags for GNU GCC (include -lgomp argument)'),
    ]

    def __init__(self, attrs=None):
        self.noopenmp = 0
        self.forceopenmp = 0
        self.noxpreprocessor = 0
        self.forcellvm = 0
        self.forcegnugcc = 0
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
    # DEPRECATED. OpenMP is now supported through -Xpreprocessor
    # if platform.system() == 'Darwin' and "CC" not in os.environ:
    #     print("Set custom environment variables for Homebrew Clang because CC is not set")
    #     cppflags = []
    #     if "CPPFLAGS" in os.environ:
    #         cppflags.append(os.environ["CPPFLAGS"])
    #     cflags = []
    #     if "CFLAGS" in os.environ:
    #         cflags.append(os.environ["CFLAGS"])
    #     ldflags = []
    #     if "LDFLAGS" in os.environ:
    #         ldflags.append(os.environ["LDFLAGS"])
    #     if os.path.exists("/usr/local/opt/llvm/bin/clang"):
    #         # We have a recent version of LLVM that probably supports openmp to compile parallel C code (installed using
    #         # `brew install llvm`).
    #         os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
    #         print("CC={}".format(os.environ["CC"]))
    #         ldflags += ["-L/usr/local/opt/llvm/lib"]
    #         cppflags += ["-I/usr/local/opt/llvm/include"]
    #         cflags += ["-I/usr/local/opt/llvm/include"]
    #         try:
    #             mac_ver = [int(nb) for nb in platform.mac_ver()[0].split(".")]
    #             if mac_ver[0] == 10 and mac_ver[1] >= 14:
    #                 # From Mojave on, the header files are part of Xcode.app
    #                 incpath = '-I/Applications/Xcode.app/Contents/Developer/Platforms/' + \
    #                           'MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include'
    #                 cppflags += [incpath]
    #                 cflags += [incpath]
    #         except Exception as exc:
    #             print("Failed to check version")
    #             print(exc)
    #     else:
    #         # The default clang in XCode is compatible with OpenMP when using -Xpreprocessor
    #         pass
    #
    #     if len(cppflags) > 0:
    #         os.environ["CPPFLAGS"] = " ".join(cppflags)
    #         print("CPPFLAGS={}".format(os.environ["CPPFLAGS"]))
    #     if len(cflags) > 0:
    #         os.environ["CFLAGS"] = " ".join(cflags)
    #         print("CFLAGS={}".format(os.environ["CFLAGS"]))
    #     if len(ldflags) > 0:
    #         os.environ["LDFLAGS"] = " ".join(ldflags)
    #         print("LDFLAGS={}".format(os.environ["LDFLAGS"]))
    # else:
    #     print("Using the following environment variables:")
    #     print("CC={}".format(os.environ.get("CC", "")))
    #     print("CPPFLAGS={}".format(os.environ.get("CPPFLAGS", "")))
    #     print("CFLAGS={}".format(os.environ.get("CPLAGS", "")))
    #     print("LDFLAGS={}".format(os.environ.get("LDFLAGS", "")))


class MyBuildExtCommand(BuildExtCommand):

    def build_extensions(self):
        c = self.compiler.compiler_type
        # Custom for homebrew
        print("Compiler type: {}".format(c))
        print("--noopenmp: {}".format(self.distribution.noopenmp))
        print("--forceopenmp: {}".format(self.distribution.forceopenmp))
        print("--noxpreprocessor: {}".format(self.distribution.noxpreprocessor))
        print("--forcellvm: {}".format(self.distribution.forcellvm))
        print("--forcegnugcc: {}".format(self.distribution.forcegnugcc))

        if self.distribution.forcellvm or (c == "unix" and ("local/opt/llvm" in self.compiler.compiler[0] or
                                                            "clang" in self.compiler.compiler[0])):
            # local/opt/llvm is homebrew
            print('Using LLVM settings ({})'.format(self.compiler.compiler[0]))
            c = 'llvm'
        elif self.distribution.forcegnugcc or \
                (c == "unix" and ("gnu-gcc" in self.compiler.compiler[0]) or
                                 ("gnu-cc" in self.compiler.compiler[0])):
            print('Using GNU GCC settings ({})'.format(self.compiler.compiler[0]))
            c = 'gnugcc'

        if self.distribution.noopenmp == 0 and self.distribution.forceopenmp == 0:
            try:
                check_result = check_openmp(self.compiler.compiler[0], self.distribution.noxpreprocessor)
            except Exception as exc:
                print("WARNING: Cannot check for OpenMP, assuming to be available")
                print(exc)
                check_result = True  # Assume to be present by default
            if not check_result:
                print("WARNING: OpenMP is not available, disabling OpenMP (no parallel computing in C)")
                self.distribution.noopenmp = 1
                # Not removing the dtw_cc_omp extension, this will be compiled but
                # without any real functionality except is_openmp_supported()
        if c in c_args:
            if self.distribution.noopenmp == 1:
                args = [arg for arg in c_args[c] if arg not in ['-Xpreprocessor', '-fopenmp', '-lomp', '-lgomp']]
            elif self.distribution.noxpreprocessor == 1:
                args = [arg for arg in c_args[c] if arg not in ['-Xpreprocessor']]
            else:
                args = c_args[c]
            for e in self.extensions:
                e.extra_compile_args = args
        else:
            print("Unknown compiler type: {}".format(c))
        if c in l_args:
            if self.distribution.noopenmp == 1:
                args = [arg for arg in l_args[c] if arg not in ['-Xpreprocessor', '-fopenmp', '-lomp']]
            elif self.distribution.noxpreprocessor == 1:
                args = [arg for arg in l_args[c] if arg not in ['-Xpreprocessor']]
            else:
                args = l_args[c]
            for e in self.extensions:
                e.extra_link_args = args
        else:
            print("Unknown linker type: {}".format(c))
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


def check_openmp(cc_bin, noxpreprocessor):
    """Check if OpenMP is available"""
    print("Checking for OpenMP availability for {}".format(cc_bin))
    cc_binname = os.path.basename(cc_bin)
    args = None
    kwargs = None
    if "clang" in cc_binname or "cc" in cc_binname:
        if noxpreprocessor == 0:
            args = [[str(cc_bin), "-dM", "-E", "-Xpreprocessor", "-fopenmp", "-"]]
        else:
            args = [[str(cc_bin), "-dM", "-E", "-fopenmp", "-"]]
        kwargs = {"stdout": sp.PIPE, "stderr": sp.PIPE, "input": '', "encoding": 'ascii'}
        print(" ".join(args[0]) + " # with " + ", ".join(str(k) + "=" + str(v) for k, v in kwargs.items()))
    if args is not None:
        try:
            p = sp.run(*args, **kwargs)
            print(p.stderr)
            defs = p.stdout.splitlines()
            for curdef in defs:
                if "_OPENMP" in curdef:
                    print(curdef)
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
    # - Cython uses the glob package to find files, thus use unix-style paths
    # - Multiple extensions are created to have a sub-package per type of distance
    #   and per functionality (e.g. with or without OpenMP).
    #   The disadvantage is that the same C-files are reused for multiple extensions
    extensions.append(
        Extension(
            "dtaidistance.dtw_cc",
            ["dtaidistance/dtw_cc.pyx", "dtaidistance/dtw_cc.pxd",
             "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_dtw.c",
             "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_ed.c"
             ],
            depends=["dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_globals.h",
                     "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_ed.h"],
            include_dirs=[str(dtaidistancec_path), "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC"],
            library_dirs=[str(dtaidistancec_path), "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC"],
            extra_compile_args=[],
            extra_link_args=[]))
    extensions.append(
        Extension(
            "dtaidistance.ed_cc",
            ["dtaidistance/ed_cc.pyx",
             "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_ed.c"],
            depends=["dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_globals.h"],
            include_dirs=[str(dtaidistancec_path), "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC"],
            extra_compile_args=[],
            extra_link_args=[]))
    extensions.append(
        Extension(
            "dtaidistance.dtw_cc_omp",
            ["dtaidistance/dtw_cc_omp.pyx",
             "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_dtw_openmp.c",
             "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_dtw.c",
             "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_ed.c"],
            depends=["dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_globals.h",
                     "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_dtw.h"
                     "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_ed.h"],
            include_dirs=[str(dtaidistancec_path), "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC"],
            extra_compile_args=[],
            extra_link_args=[]))

    if numpy is not None:
        extensions.append(
            Extension(
                "dtaidistance.dtw_cc_numpy", ["dtaidistance/util_numpy_cc.pyx"],
                depends=["dtaidistance/lib/DTAIDistanceC/DTAIDistanceC/dd_globals.h"],
                include_dirs=[numpy.get_include(), str(dtaidistancec_path), "dtaidistance/lib/DTAIDistanceC/DTAIDistanceC"],
                extra_compile_args=[],
                extra_link_args=[]))
    else:
        print("WARNING: Numpy was not found, preparing a version without Numpy support.")

    ext_modules = cythonize(extensions)
                            # compiler_directives={'language_level': "3"})
else:
    print("WARNING: Cython was not found, preparing a pure Python version.")
    ext_modules = []

# It is easier to include numpy because of the build isolation (PEP517), even
# though it is optional. Otherwise the --no-build-isolation flag would be required.
# If you want to ignore numpy, remove it here and in pyproject.toml.
install_requires = ['numpy']  # 'cython>=0.29.6',
setup_requires = ['numpy']  # 'setuptools>=18.0', 'cython>=0.29.6',
tests_require = ['pytest', 'pytest-benchmark']

# Check version number
init_fn = here / 'dtaidistance' / '__init__.py'
with init_fn.open('r', encoding='utf-8') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)
if not version:
    raise RuntimeError('Cannot find version information')

# Set up readme file
readme_path = here / 'README.md'
if os.path.exists(readme_path):
    with readme_path.open('r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = ""
long_description_content_type = "text/markdown"

# Create setup
setup_kwargs = {}
def set_setup_kwargs(**kwargs):
    global setup_kwargs
    setup_kwargs = kwargs

set_setup_kwargs(
    name='dtaidistance',
    version=version,
    description='Distance measures for time series (Dynamic Time Warping, fast C implementation)',
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    author='Wannes Meert',
    author_email='wannes.meert@cs.kuleuven.be',
    url='https://github.com/wannesm/dtaidistance',
    project_urls={
        'DTAIDistance documentation': 'http://dtaidistance.readthedocs.io/en/latest/',
        'DTAIDistance source': 'https://github.com/wannesm/dtaidistance'
    },
    packages=['dtaidistance', 'dtaidistance.clustering', 'dtaidistance.subsequence'],
    python_requires='>=3.5',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={
        'vis': ['matplotlib'],
        'numpy': ['numpy', 'scipy'],
        'all': ['matplotlib', 'numpy', 'scipy']
    },
    include_package_data=True,
    package_data={
        'dtaidistance': ['*.pyx', '*.pxd', '*.c', '*.h'],
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
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
    ],
    keywords='dtw, time series, dynamic time warping, distance',
    zip_safe=False
)

try:
    setup(ext_modules=ext_modules, **setup_kwargs)
except (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, SystemExit) as exc:
    print("ERROR: The C extension could not be compiled")
    print(exc)

    if 'build_ext' in setup_kwargs['cmdclass']:
        del setup_kwargs['cmdclass']['build_ext']

    setup(**setup_kwargs)
    print("Installed the plain Python version of the package.")
    print("If you need the C extension, try reinstalling.")
