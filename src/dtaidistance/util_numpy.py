from contextlib import ContextDecorator
import os
import logging
import importlib

from .exceptions import NumpyException, ScipyException, PandasException


logger = logging.getLogger("be.kuleuven.dtai.distance")


def test_without_numpy():
    if "DTAIDISTANCE_TESTWITHOUTNUMPY" in os.environ and os.environ["DTAIDISTANCE_TESTWITHOUTNUMPY"] == "1":
        return True
    return False


def test_without_scipy():
    if test_without_numpy():
        return True
    if "DTAIDISTANCE_TESTWITHOUTSCIPY" in os.environ and os.environ["DTAIDISTANCE_TESTWITHOUTSCIPY"] == "1":
        return True
    return False


def test_without_pandas():
    if test_without_numpy():
        return True
    if "DTAIDISTANCE_TESTWITHOUTPANDAS" in os.environ and os.environ["DTAIDISTANCE_TESTWITHOUTPANDAS"] == "1":
        return True
    return False


def verify_np_array(seq):
    try:
        np = importlib.import_module("numpy")
    except ImportError as e:
        raise NumpyException("Expects numpy to be installed") from e
    if np is not None:
        if isinstance(seq, (np.ndarray, np.generic)):
            if not seq.data.c_contiguous:
                logger.debug("Warning: Sequence 1 passed to method distance is not C-contiguous. " +
                             "The sequence will be copied.")
                seq = seq.copy(order='C')
    return seq


class test_uses_numpy(ContextDecorator):
    def __init__(self, strict=True):
        """Context to construct tests that use the optional dependency Numpy.

        :param strict: Throw error if Numpy is not used (to remove context where not necessary)
        :return: Numpy stub
        """
        self.strict = strict
        self.testwithoutnp = test_without_numpy()
        self.e = None
        try:
            self.np = importlib.import_module("numpy")
        except (ImportError, ValueError) as e:
            self.np = None
            self.e = e

    def __enter__(self):
        if self.testwithoutnp or self.np is None:
            raise NumpyException("Numpy excepted to be available for test. "
                                 "Set DTAIDISTANCE_TESTWITHOUTNUMPY=1 to test without Numpy.") from self.e
        return self.np

    def __exit__(self, *exc):
        if self.testwithoutnp:
            if exc[0] is None:
                if self.strict and self.testwithoutnp:
                    # If no NumpyException is thrown, this test did not use Numpy because no error was thrown
                    # and should not use this decorator
                    raise Exception("Test does not use Numpy, remove decorator!")
                else:
                    return
            if issubclass(exc[0], NumpyException):
                return True


class test_uses_scipy(ContextDecorator):
    def __init__(self, strict=True):
        """Context to construct tests that use the optional dependency Scipy.

        :param strict: Throw error if Scipy is not used (to remove context where not necessary)
        :return: Numpy stub
        """
        self.strict = strict
        self.testwithoutscipy = test_without_scipy()
        try:
            self.scipy = importlib.import_module("scipy")
        except ImportError as _e:
            self.scipy = None

    def __enter__(self):
        if self.testwithoutscipy or self.scipy is None:
            raise ScipyException("Scipy excepted to be available for test. "
                                 "Set DTAIDISTANCE_TESTWITHOUTSCIPY=1 to test without Scipy.")
        return self.scipy

    def __exit__(self, *exc):
        if self.testwithoutscipy:
            if exc[0] is None:
                if self.strict and self.testwithoutscipy:
                    # If no ScipyException is thrown, this test did not use Scipy because no error was thrown
                    # and should not use this decorator
                    raise Exception("Test does not use Scipy, remove decorator!")
                else:
                    return
            if issubclass(exc[0], ScipyException):
                return True


class test_uses_pandas(ContextDecorator):
    def __init__(self, strict=True):
        """Context to construct tests that use the optional dependency Pandas.

        :param strict: Throw error if Pandas is not used (to remove context where not necessary)
        :return: Numpy stub
        """
        self.strict = strict
        self.testwithoutpandas = test_without_pandas()
        self.e = None
        try:
            self.pandas = importlib.import_module("pandas")
        except (ImportError, ValueError) as e:
            self.pandas = None
            self.e = e

    def __enter__(self):
        if self.testwithoutpandas or self.pandas is None:
            raise PandasException("Pandas excepted to be available for test. "
                                  "Set DTAIDISTANCE_TESTWITHOUTPANDAS=1 to test without Pandas.") from self.e
        return self.pandas

    def __exit__(self, *exc):
        if self.testwithoutpandas:
            if exc[0] is None:
                if self.strict and self.testwithoutpandas:
                    # If no PandasException is thrown, this test did not use Pandas because no error was thrown
                    # and should not use this decorator
                    raise Exception("Test does not use Pandas, remove decorator!")
                else:
                    return
            if issubclass(exc[0], PandasException):
                return True
