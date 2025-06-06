#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
 python3 setup.py build_ext --inplace
"""
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as BuildExtCommand
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
    import Cython
    from Cython.Build import cythonize
except ImportError:
    Cython = None
    cythonize = None

here = Path(__file__).parent
dtaidistancec_path = Path('src') / 'DTAIDistanceC' / 'DTAIDistanceC'

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


class MyDistribution(Distribution):
    global_options = Distribution.global_options + [
        ('noopenmp', None, 'No compiler/linker flags for OpenMP (OpenMP disabled)'),
        ('forceopenmp', None, 'Force compiler/linker flags to use OpenMP'),
        ('noxpreprocessor', None, 'Assume OpenMP is built-in (remove -Xpreprocessor argument)'),
        ('forcellvm', None, 'Force compile/linker flags for LLVM (include -lomp argument)'),
        ('forcegnugcc', None, 'Force compile/linker flags for GNU GCC (include -lgomp argument)'),
        ('forcestatic', None, 'Try to force the linker to use the static OMP library.')
    ]

    def __init__(self, attrs=None):
        self.noopenmp = 0
        self.forceopenmp = 0
        self.noxpreprocessor = 0
        self.forcellvm = 0
        self.forcegnugcc = 0
        self.forcestatic = 0
        super().__init__(attrs)



# def set_custom_envvars_for_homebrew():
#     """Update environment variables automatically for Homebrew if CC is not set"""
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
        """Provide support for compiling with OpenMP support (for parellellization).

        This seems not to be supported out-of-the-box by Cython, which also requires
        to set the openmp compiler flags as part of the setup.py file:
        https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html#compiling
        """
        try:
            fp = open(os.path.join(self.build_lib, 'dtaidistance', 'compilation.log'), 'w')
        except Exception as exc:
            print('Could not open compilation.log file')
            print(exc)
            fp = None

        def print2(*args, **kwargs):
            """Print to terminal and to log file."""
            print(*args, **kwargs)
            if fp is not None:
                print(*args, file=fp, **kwargs)

        c = self.compiler.compiler_type
        print2("Compiler type: {}".format(c))
        print2("--noopenmp: {}".format(self.distribution.noopenmp))
        print2("--forceopenmp: {}".format(self.distribution.forceopenmp))
        print2("--noxpreprocessor: {}".format(self.distribution.noxpreprocessor))
        print2("--forcellvm: {}".format(self.distribution.forcellvm))
        print2("--forcegnugcc: {}".format(self.distribution.forcegnugcc))
        print2("--forcestatic: {}".format(self.distribution.forcestatic))

        # Check cython information
        if Cython is None:
            print2("Cython package not found")
        else:
            print2("Cython found (during compilation)")
            print2("- Version: {}".format(Cython.__version__))
            print2("- Locations: {}".format(Cython))

        # Check numpy information
        if numpy is None:
            print2("Numpy package not found")
        else:
            print2("Numpy found (during compilation):")
            print2("- Version: {}".format(numpy.__version__))
            print2("- Location: {}".format(numpy))

        if c == "unix" and "cc" in self.compiler.compiler[0]:
            gcc_is_clang = check_clang(self.compiler.compiler[0], printfn=print2)
        else:
            gcc_is_clang = False

        if self.distribution.forcellvm or gcc_is_clang or \
                (c == "unix" and ("llvm" in self.compiler.compiler[0] or
                                  "clang" in self.compiler.compiler[0])):
            # Homebrew:
            # /usr/local/opt/llvm is homebrew
            # For homebrew it is assumed that following paths are set systemwide:
            # /usr/local/opt/llvm/bin/clang -I/usr/local/opt/llvm/include -L/usr/local/opt/llvm/lib
            # macOS:
            # http://blog.llvm.org/2015/05/openmp-support_22.html
            # https://www.mathworks.com/help/coder/ug/install-openmp-library-on-macos-platform.html
            print2('Using LLVM settings ({})'.format(self.compiler.compiler[0]))
            c = 'llvm'
        elif self.distribution.forcegnugcc or \
                (c == "unix" and (("gcc" in self.compiler.compiler[0]) or
                                  ("gnu-cc" in self.compiler.compiler[0]))):
            print2('Using GNU GCC settings ({})'.format(self.compiler.compiler[0]))
            c = 'gnugcc'

        if self.distribution.noopenmp == 0 and self.distribution.forceopenmp == 0:
            # See which paths for libraries exist to add to compiler
            # MacPorts
            p = Path('/usr/local/opt/libomp/include')
            if p.exists():
                print(f'Adding path to compiler {p}')
                c_args['unix'].append(f'-I{p}')
                c_args['llvm'].append(f'-I{p}')
            p = Path('/opt/local/lib/libomp')
            if p.exists():
                print(f'Adding path to linker {p}')
                l_args['unix'].append(f'-L{p}')
                l_args['llvm'].append(f'-L{p}')
            # HomeBrew
            #p = Path('/opt/homebrew/include')
            p = Path('/opt/homebrew/opt/libomp/include') # Location changed
            if p.exists():
                print(f'Adding path to compiler: {p}')
                c_args['unix'].append(f'-I{p}')
                c_args['llvm'].append(f'-I{p}')
            #p = Path('/opt/homebrew/lib')
            p = Path('/opt/homebrew/opt/libomp/lib') # Location changed
            if p.exists():
                libomp = Path(p / 'libomp.a')
                if self.distribution.forcestatic and platform.system() == 'Darwin' and libomp.exists():
                        # Force the linker to use the static libomp.a library
                        # This is useful to create wheels for people who do not
                        # have the shared library libomp.{dylib,so} in the right location
                        print(f'Removing -lomp and adding libomp to linker: {libomp}')
                        l_args['unix'] = [a for a in l_args['unix'] if a != '-lomp']
                        l_args['llvm'] = [a for a in l_args['llvm'] if a != '-lomp']
                        l_args['unix'].append(str(libomp))
                        l_args['llvm'].append(str(libomp))
                else:
                    print(f'Adding path to linker: {p}')
                    l_args['unix'].append(f'-L{p}')
                    l_args['llvm'].append(f'-L{p}')

            # Try to check availability of OpenMP
            try:
                check_result = check_openmp(self.compiler.compiler[0], self.distribution.noxpreprocessor,
                                            printfn=print2)
            except Exception as exc:
                print2("WARNING: Cannot check for OpenMP, assuming to be available")
                print2(exc)
                check_result = True  # Assume to be present by default
            if not check_result:
                print2("WARNING: OpenMP is not available, disabling OpenMP (no parallel computing in C)")
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
            print2("Unknown compiler type: {}".format(c))
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
            print2("Unknown linker type: {}".format(c))
        if numpy is None:
            self.extensions = [arg for arg in self.extensions if "numpy" not in str(arg)]
        print2(f'All extensions:')
        print2(self.extensions)
        if fp is not None:
            try:
                fp.close()
            except Exception as exc:
                print('Could not close compilation.log')
                print(exc)

        BuildExtCommand.build_extensions(self)

    def initialize_options(self):
        # set_custom_envvars_for_homebrew()
        super().initialize_options()

    # def finalize_options(self):
    #     super().finalize_options()

    # def run(self):
    #     super().run()


class MyBuildExtInPlaceCommand(MyBuildExtCommand):
    def initialize_options(self):
        super().initialize_options()
        self.inplace = True


def check_clang(cc_bin, printfn=print):
    """Check if gcc is really an xcrun to clang"""
    printfn("Checking if {} redirects to clang".format(cc_bin))
    args = [[str(cc_bin), "--version"]]
    kwargs = {"stdout": sp.PIPE, "stderr": sp.PIPE, "input": '', "encoding": 'ascii'}
    printfn(" ".join(args[0]) + " # with " + ", ".join(str(k) + "=" + str(v) for k, v in kwargs.items()))
    try:
        p = sp.run(*args, **kwargs)
        printfn(p.stderr)
        defs = p.stdout.splitlines()
        for curdef in defs:
            if "clang" in curdef:
                printfn(curdef)
                printfn("... found clang")
                return True
    except Exception:
        printfn("... no clang")
        return False
    return False


def check_openmp(cc_bin, noxpreprocessor, printfn=print):
    """Check if OpenMP is available"""
    printfn("Checking for OpenMP availability for {}".format(cc_bin))
    cc_binname = os.path.basename(cc_bin)
    args = None
    kwargs = None
    if "clang" in cc_binname or "cc" in cc_binname:
        if noxpreprocessor == 0:
            args = [[str(cc_bin), "-dM", "-E", "-Xpreprocessor", "-fopenmp", "-"]]
        else:
            args = [[str(cc_bin), "-dM", "-E", "-fopenmp", "-"]]
        kwargs = {"stdout": sp.PIPE, "stderr": sp.PIPE, "input": '', "encoding": 'ascii'}
        printfn(" ".join(args[0]) + " # with " + ", ".join(str(k) + "=" + str(v) for k, v in kwargs.items()))
    if args is not None:
        try:
            p = sp.run(*args, **kwargs)
            printfn(p.stderr)
            defs = p.stdout.splitlines()
            for curdef in defs:
                if "_OPENMP" in curdef:
                    printfn(curdef)
                    printfn("... found OpenMP")
                    return True
        except Exception:
            printfn("... no OpenMP")
            return False
    else:
        printfn("... do not know how to check for OpenMP (unknown CC), assume to be available")
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
            ["src/dtaidistance/dtw_cc.pyx",
             "src/DTAIDistanceC/DTAIDistanceC/dd_dtw.c",
             "src/DTAIDistanceC/DTAIDistanceC/dd_ed.c",
             "src/DTAIDistanceC/DTAIDistanceC/dd_globals.c"
             ],
            depends=["src/DTAIDistanceC/DTAIDistanceC/dd_globals.h",
                     "src/DTAIDistanceC/DTAIDistanceC/dd_ed.h"],
            include_dirs=[str(dtaidistancec_path),
                          "src/DTAIDistanceC/DTAIDistanceC"],
            library_dirs=[str(dtaidistancec_path),
                          "src/DTAIDistanceC/DTAIDistanceC"],
            extra_compile_args=[],
            extra_link_args=[]))
    extensions.append(
        Extension(
            "dtaidistance.ed_cc",
            ["src/dtaidistance/ed_cc.pyx",
             "src/DTAIDistanceC/DTAIDistanceC/dd_ed.c",
             "src/DTAIDistanceC/DTAIDistanceC/dd_globals.c"],
            depends=["src/DTAIDistanceC/DTAIDistanceC/dd_globals.h"],
            include_dirs=[str(dtaidistancec_path),
                          "src/DTAIDistanceC/DTAIDistanceC"],
            extra_compile_args=[],
            extra_link_args=[]))
    extensions.append(
        Extension(
            "dtaidistance.dtw_cc_omp",
            ["src/dtaidistance/dtw_cc_omp.pyx",
             "src/DTAIDistanceC/DTAIDistanceC/dd_dtw_openmp.c",
             "src/DTAIDistanceC/DTAIDistanceC/dd_dtw.c",
             "src/DTAIDistanceC/DTAIDistanceC/dd_ed.c",
             "src/DTAIDistanceC/DTAIDistanceC/dd_globals.c"],
            depends=["src/DTAIDistanceC/DTAIDistanceC/dd_globals.h",
                     "src/DTAIDistanceC/DTAIDistanceC/dd_dtw.h",
                     "src/DTAIDistanceC/DTAIDistanceC/dd_ed.h"],
            include_dirs=[str(dtaidistancec_path),
                          "src/DTAIDistanceC/DTAIDistanceC"],
            extra_compile_args=[],
            extra_link_args=[]))

    if numpy is not None:
        extensions.append(
            Extension(
                "dtaidistance.dtw_cc_numpy",
                ["src/dtaidistance/util_numpy_cc.pyx",
                 "src/DTAIDistanceC/DTAIDistanceC/dd_globals.c"],
                depends=["src/DTAIDistanceC/DTAIDistanceC/dd_globals.h"],
                include_dirs=[numpy.get_include(),
                              str(dtaidistancec_path),
                              "src/DTAIDistanceC/DTAIDistanceC"],
                extra_compile_args=[],
                extra_link_args=[]))
    else:
        print("WARNING: Numpy was not found, preparing a version without Numpy support.")

    ext_modules = cythonize(extensions, language_level=2)

else:
    print("WARNING: Cython was not found, preparing a pure Python version.")
    ext_modules = []


# Create setup
setup_kwargs = {}
def set_setup_kwargs(**kwargs):
    global setup_kwargs
    setup_kwargs = kwargs

set_setup_kwargs(
    distclass=MyDistribution,
    cmdclass={
        'buildinplace': MyBuildExtInPlaceCommand,
        'build_ext': MyBuildExtCommand,
    },
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

