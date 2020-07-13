# -*- coding: UTF-8 -*-
"""
dtaidistance.util
~~~~~~~~~~~~~~~~~

Utility functions for DTAIDistance.

:author: Wannes Meert
:copyright: Copyright 2017-2018 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import os
import sys
import csv
import logging
from array import array
from pathlib import Path
import tempfile


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


logger = logging.getLogger("be.kuleuven.dtai.distance")


dtaidistance_dir = os.path.abspath(os.path.dirname(__file__))


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


class SeriesContainer:
    def __init__(self, series):
        """Container for a list of series.

        This wrapper class knows how to deal with multiple types of datastructures to represent
        a list of sequences:
        - List[array.array]
        - List[numpy.array]
        - List[List]
        - numpy.array
        - numpy.matrix

        When using the C-based extensions, the data is automatically verified and converted.
        """
        if isinstance(series, SeriesContainer):
            self.series = series.series
        elif np is not None and isinstance(series, np.ndarray) and len(series.shape) == 2:
            # A matrix always returns a 2D array, also if you select one row (to be consistent
            # and always be a matrix datastructure). The methods in this toolbox expect a
            # 1D array thus we need to convert to a 1D or 2D array.
            # self.series = [np.asarray(series[i]).reshape(-1) for i in range(series.shape[0])]
            self.series = np.asarray(series, order="C")
        elif type(series) == set or type(series) == tuple:
            self.series = list(series)
        else:
            self.series = series

    def c_data(self):
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

    def __getitem__(self, item):
        return self.series[item]

    def __len__(self):
        return len(self.series)

    def __str__(self):
        return "SeriesContainer:\n{}".format(self.series)

    @staticmethod
    def wrap(series):
        if isinstance(series, SeriesContainer):
            return series
        return SeriesContainer(series)


def recompile():
    import subprocess as sp

    sp.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=dtaidistance_dir)


def argmin(a):
    imin, vmin = 0, float("inf")
    for i, v in enumerate(a):
        if v < vmin:
            imin, vmin = i, v
    return imin
