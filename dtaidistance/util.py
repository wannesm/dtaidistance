"""
dtaidistance.util - Auxiliary methods for Time Series

__author__ = "Wannes Meert"
__copyright__ = "Copyright 2018 KU Leuven, DTAI Research Group"
__license__ = "APL"

..
    Part of the DTAI distance code.

    Copyright 2018 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import os
import logging
from array import array
from pathlib import Path
import numpy as np


logger = logging.getLogger("be.kuleuven.dtai.distance")


dtaidistance_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)


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
        if isinstance(series, np.matrix):
            # A matrix always returns a 2D array, also if you select one row (to be consistent
            # and always be a matrix datastructure). The methods in this toolbox expect a
            # 1D array thus we need to convert to a 1D or 2D array.
            # self.series = [np.asarray(series[i]).reshape(-1) for i in range(series.shape[0])]
            self.series = np.asarray(series, order='C')
        elif type(series) == set:
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
        if type(self.series) == list:
            for i in range(len(self.series)):
                serie = self.series[i]
                if isinstance(serie, np.ndarray):
                    if not serie.flags.c_contiguous:
                        serie = np.asarray(serie, order='C')
                        self.series[i] = serie
                elif isinstance(serie, array):
                    pass
                else:
                    raise Exception("Type of series not supported, "
                                    f"expected numpy.array or array.array but got {type(serie)}")
        elif isinstance(self.series, np.ndarray):
            if not self.series.flags.c_contiguous:
                self.series = self.series.copy(order='C')
        return self.series

    def __getitem__(self, item):
        return self.series[item]

    def __len__(self):
        return len(self.series)

    def __str__(self):
        return f"SeriesContainer:\n{self.series}"

    @staticmethod
    def wrap(series):
        if isinstance(series, SeriesContainer):
            return series
        return SeriesContainer(series)
