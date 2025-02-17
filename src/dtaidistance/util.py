# -*- coding: UTF-8 -*-
"""
dtaidistance.util
~~~~~~~~~~~~~~~~~

Utility functions for DTAIDistance.

:author: Wannes Meert
:copyright: Copyright 2017-2022 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import os
import sys
import csv
import logging
from array import array
from pathlib import Path
import tempfile


logger = logging.getLogger("be.kuleuven.dtai.distance")


try:
    import numpy as np
except ImportError:
    np = None

try:
    from . import dtw_cc
except ImportError:
    dtw_cc = None

try:
    from . import dtw_cc_omp
except ImportError:
    dtw_cc_omp = None

try:
    from . import dtw_cc_numpy
except ImportError:
    dtw_cc_numpy = None
except ValueError as exc:
    logger.warning('Warning: loading library to link with numpy returned an error')
    logger.warning(exc)
    dtw_cc_numpy = None


dtaidistance_dir = os.path.abspath(os.path.dirname(__file__))


def try_import_c(verbose=False):
    is_complete = True
    msgs = []
    global dtw_cc
    global dtw_cc_omp
    global dtw_cc_numpy
    try:
        from . import dtw_cc
    except Exception as exc:
        print('Cannot import C-based library (dtw_cc)')
        msgs.append('Cannot import C-based library (dtw_cc)')
        msgs.append(str(exc))
        dtw_cc = None
        is_complete = False
    try:
        from . import dtw_cc_omp
    except Exception as exc:
        print('Cannot import OMP-based library (dtw_cc_omp)')
        msgs.append('Cannot import OMP-based library (dtw_cc_omp)')
        msgs.append(str(exc))
        dtw_cc_omp = None
        is_complete = False
    try:
        from . import dtw_cc_numpy
    except Exception as exc:
        print('Cannot import Numpy-based library (dtw_cc_numpy)')
        msgs.append('Cannot import Numpy-based library (dtw_cc_numpy)')
        msgs.append(str(exc))
        dtw_cc_numpy = None
        is_complete = False
    try:
        import numpy
        msgs.append('Numpy version: {}'.format(numpy.__version__))
    except Exception as exc:
        print('Cannot import Numpy (optional dependency)')
        msgs.append('Cannot import Numpy (optional dependency)')
        msgs.append(str(exc))
    try:
        import matplotlib
        msgs.append('Matplotlib version: {}'.format(matplotlib.__version__))
    except Exception as exc:
        print('Cannot import Matplotlib (optional dependency)')
        msgs.append('Cannot import Matplotlib (optional dependency)')
        msgs.append(str(exc))
    try:
        import scipy
        msgs.append('Scipy version: {}'.format(scipy.__version__))
    except Exception as exc:
        print('Cannot import SciPy (optional dependency)')
        msgs.append('Cannot import SciPy (optional dependency)')
        msgs.append(str(exc))
    if not is_complete:
        print('\nNot all libraries are available in your installation. ')
        print('You can rerun the compilation from source or pip install in verbose mode:\n'
              'pip install -vvv --upgrade --force-reinstall --no-deps --no-binary dtaidistance dtaidistance')
        print('In case you need to use an older version of numpy, compile against your current installation:\n'
              'pip install -vvv --upgrade --force-reinstall --no-deps --no-build-isolation '
              '--no-binary dtaidistance dtaidistance')
        print('\nShare the following information when submitting a bug report:')
    elif verbose:
        print('All ok ...')
    if not is_complete or verbose:
        print('== Packages ==')
        for msg in msgs:
            print('- {}'.format(msg))
        print('== System information ==')
        import sys
        print(sys.implementation)
        print('== Compilation information ==')
        try:
            import pkgutil
            logtxt = pkgutil.get_data(__name__, "compilation.log")
            print(logtxt.decode())
        except Exception as exc:
            print('Could not read compilation.log')
            print(exc)
        print('')
        print('==')
    return is_complete


def prepare_directory(directory=None):
    """Prepare the given directory, create it if necessary.
    If no directory is given, a new directory will be created in the system's temp directory.
    """
    if directory is not None:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True)
        logger.debug("Using directory: {}".format(directory))
        return Path(directory)
    directory = tempfile.mkdtemp(prefix="dtaidistance_")
    logger.debug("Using directory: {}".format(directory))
    return Path(directory)


def read_substitution_matrix(file):
    """Read substitution matrix from file.

    Comments starting with # and newlines are allowed anywhere
    in the file.
    
    :return: A dictionary mapping tuples of symbols to their weight.
    """

    def strip_comments(reader):
        for line in reader:
            if not line.rstrip() or line[0] == '#':
                continue
            yield line.rstrip()

    matrix = dict()
    with open(file) as f:
        reader = csv.reader(strip_comments(f), delimiter=" ", skipinitialspace=True)
        line = next(reader)
        idx = {i: symbol for i, symbol in enumerate(line)}
        for line in reader:
            symbol = line[0]
            for j, value in enumerate(line[1:]):
                matrix[(idx[j], symbol)] = float(value)
    return matrix


def detect_ndim(s):
    if np is not None and isinstance(s, np.ndarray):
        return s.ndim
    if type(s) is list and len(s) > 0:
        return detect_ndim(s[0]) + 1
    if type(s) in [int, float]:
        return 0
    return None


class SeriesContainer:
    def __init__(self, series, support_ndim=True):
        """Container for a list of series.

        This wrapper class knows how to deal with multiple types of datastructures to represent
        a sequence of sequences.

        For 1 dimensional time series (e.g. [1,2,3,4]):
        - List[array.array]
        - List[numpy.array] (array is 1 dimensional)
        - List[List]
        - numpy.array (2 dimensional)
        - numpy.matrix

        For n-dimensional time series (e.g. [[1,2],[3,4],[5,6]]):
        - List[numpy.array] (array is 2 dimensional)
        - List[List[List]]
        - numpy.array (3 dimensional)

        When using the C-based extensions, the data is automatically verified and converted.
        """
        self.support_ndim = support_ndim
        # Always detect the dimensionality of the time series, even if support_ndim is false
        self.detected_ndim = False
        if isinstance(series, SeriesContainer):
            self.series = series.series
        elif np is not None and isinstance(series, np.ndarray):
            # A np.matrix always returns a 2D array, also if you select one row (to be consistent
            # and always be a matrix datastructure). The methods in this toolbox expect a
            # 1D array thus we need to convert to a 1D or 2D array. This is taken care by asarray
            self.series = np.asarray(series, order="C")
            if self.series.ndim > 2:
                if not self.support_ndim:
                    raise Exception('N-dimensional series are not supported '
                                    '(series.ndim = {}) > 2'.format(self.series.ndim))
                self.detected_ndim = len(self.series[0, 0])
            else:
                self.detected_ndim = 1
        elif type(series) in [set, tuple, list]:
            self.series = list(series)
            if np is not None and isinstance(self.series[0], np.ndarray):
                if self.series[0].ndim > 1:
                    if not self.support_ndim:
                        raise Exception('N-dimensional series are not supported '
                                        '(series[0].ndim = {}) > 1'.format(self.series[0].ndim))
                    self.detected_ndim = len(self.series[0][0])
                else:
                    self.detected_ndim = 1
            elif type(series[0]) in [list, tuple]:
                if type(series[0][0]) in [list, tuple]:
                    self.detected_ndim = len(self.series[0][0])
                else:
                    self.detected_ndim = 1
        else:
            self.series = series

    def set_detected_ndim(self, ndim):
        if ndim is None:
            return
        self.detected_ndim = ndim

    def c_data_compat(self):
        """Return a datastructure that the C-component knows how to handle.
        The method tries to avoid copying or reallocating memory.

        :return: Either a list of buffers or a two-dimensional buffer. The
            buffers are guaranteed to be C-contiguous and can thus be used
            as regular pointer-based arrays in C.
        """
        if dtw_cc is None:
            raise Exception('C library not loaded')
        if type(self.series) == list:
            for i in range(len(self.series)):
                serie = self.series[i]
                if np is not None and isinstance(serie, np.ndarray):
                    if not self.support_ndim and serie.ndim != 1:
                        raise Exception('N-dimensional arrays are not supported (serie.ndim = {})'.format(serie.ndim))
                    if not serie.flags.c_contiguous:
                        serie = np.asarray(serie, order="C")
                        self.series[i] = serie
                elif isinstance(serie, array):
                    pass
                else:
                    raise Exception(
                        "Type of series not supported, "
                        "expected numpy.array or array.array but got {}".format(
                            type(serie)
                        )
                    )
            return dtw_cc.dtw_series_from_data(self.series)
        elif np is not None and isinstance(self.series, np.ndarray):
            if not self.series.flags.c_contiguous:
                logger.warning("Numpy array not C contiguous, copying data.")
                self.series = self.series.copy(order="C")
            if not self.support_ndim and self.series.ndim > 2:
                raise Exception(f'N-dimensional series are not supported (series.ndim = {self.series.ndim})')
            if dtw_cc_numpy is None:
                logger.warning("DTAIDistance C-extension for Numpy is not available. Proceeding anyway.")
                return dtw_cc.dtw_series_from_data(self.series)
            elif len(self.series.shape) == 3:
                return dtw_cc_numpy.dtw_series_from_numpy_ndim(self.series)
            else:
                return dtw_cc_numpy.dtw_series_from_numpy(self.series)
        return dtw_cc.dtw_series_from_data(self.series)

    def get_max_y(self):
        max_y = 0
        if isinstance(self.series, np.ndarray) and len(self.series.shape) == 2:
            max_y = max(np.max(self.series), abs(np.min(self.series)))
        else:
            for serie in self.series:
                max_y = max(max_y, np.max(serie), abs(np.min(serie)))
        return max_y

    def get_max_min_y(self):
        min_y, max_y = 0, 0
        if isinstance(self.series, np.ndarray) and len(self.series.shape) == 2:
            min_y = np.min(self.series)
            max_y = np.max(self.series)
        else:
            for serie in self.series:
                min_y = min(min_y, np.min(serie))
                max_y = max(max_y, np.max(serie))
        return min_y, max_y

    def get_max_length(self):
        max_length = 0
        if isinstance(self.series, np.ndarray) and len(self.series.shape) == 2:
            max_length = self.series.shape[1]
        else:
            for serie in self.series:
                max_length = max(max_length, len(serie))
        return max_length

    def get_avg_length(self):
        max_length = 0
        if isinstance(self.series, np.ndarray) and len(self.series.shape) == 2:
            max_length = self.series.shape[1]
        else:
            for serie in self.series:
                max_length += len(serie)
            max_length /= len(self.series)
        return max_length

    def __getitem__(self, item):
        return self.series[item]

    def __len__(self):
        return len(self.series)

    def __str__(self):
        return "SeriesContainer:\n{}".format(self.series)

    @staticmethod
    def wrap(series, support_ndim=True):
        if isinstance(series, SeriesContainer):
            series.support_ndim = support_ndim
            return series
        return SeriesContainer(series, support_ndim=support_ndim)


def recompile():
    import subprocess as sp

    sp.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=dtaidistance_dir)


def argmin(a):
    imin, vmin = 0, float("inf")
    for i, v in enumerate(a):
        if v < vmin:
            imin, vmin = i, v
    return imin


def argmax(a):
    imax, vmax = 0, float("-inf")
    for i, v in enumerate(a):
        if v > vmax:
            imax, vmax = i, v
    return imax


class DetectKnee:
    def __init__(self, alpha=0.3):
        """Exponential Weighted Moving Average (EWMA) based knee detection.

        Useful to detect when values start increasing at an increased rate.

        Based on:
        https://cseweb.ucsd.edu//~snoeren/papers/plush-usenix06.pdf

        :param alpha: EWMA parameter, in [0,1]
            Low values prefer old values, high values prefer recent values.
        """
        self.cnt = 0  # Number of data points seen
        self.min_points = 3  # Minimal number of data points to see before stopping
        self.arrvar_fraction = 4
        self.alpha = alpha  # EWMA parameter
        self.arr = None
        self.arrvar = None
        self.max_thr = None

    def dostop(self, value):
        if self.arr is None:
            self.arr = value
            self.arrvar = 0
            return False

        rvalue = False
        self.max_thr = self.arr + self.arrvar_fraction * self.arrvar
        # We need to see at least min_points instances to compute a reasonable arrvar
        if self.cnt >= self.min_points and value > self.max_thr:
            rvalue = True

        self.arrvar = self.alpha * max(0, value - self.arr) + (1.0 - self.alpha) * self.arrvar
        self.arr = self.alpha * value + (1.0 - self.alpha) * self.arr
        self.cnt += 1
        return rvalue
