from contextlib import ContextDecorator
import os
import logging

from .exceptions import NumpyException


logger = logging.getLogger("be.kuleuven.dtai.distance")


try:
    import numpy as np
except ImportError:
    np = None


def test_without_numpy():
    if "DTAIDISTANCE_TESTWITHOUTNUMPY" in os.environ and os.environ["DTAIDISTANCE_TESTWITHOUTNUMPY"] == "1":
        return True
    return False


def verify_np_array(seq):
    if np is not None:
        if isinstance(seq, (np.ndarray, np.generic)):
            if not seq.data.c_contiguous:
                logger.debug("Warning: Sequence 1 passed to method distance is not C-contiguous. " +
                             "The sequence will be copied.")
                seq = seq.copy(order='C')
    return seq


class NumpyStub:
    def __init__(self, testwithoutnp):
        self.testwithoutnp = testwithoutnp

    def __getattr__(self, name):
        if self.testwithoutnp or np is None:
            raise NumpyException("Numpy excepted to be available for test. "
                                 "Set DTAIDISTANCE_TESTWITHOUTNUMPY=1 to test without Numpy.")
        return getattr(np, name)


class test_uses_numpy(ContextDecorator):
    def __init__(self, strict=True):
        """Context to construct tests that use the optional dependency Numpy.

        :param strict: Throw error if Numpy is not used (to remove context where not necessary)
        :return: Numpy stub
        """
        self.strict = strict
        self.testwithoutnp = test_without_numpy()

    def __enter__(self):
        return NumpyStub(self.testwithoutnp)

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
