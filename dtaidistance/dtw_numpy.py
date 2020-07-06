from contextlib import ContextDecorator
import os

from .exceptions import NumpyException


try:
    import numpy as np
except ImportError:
    np = None


def test_without_numpy():
    if "DTAIDISTANCE_TESTWITHOUTNUMPY" in os.environ and os.environ["DTAIDISTANCE_TESTWITHOUTNUMPY"] == "1":
        return True
    return False


class NumpyStub:
    def __init__(self, testwithoutnp):
        self.testwithoutnp = testwithoutnp

    def __getattr__(self, name):
        if self.testwithoutnp or np is None:
            raise NumpyException("Numpy excepted to be available for test. "
                                 "Set DTAIDISTANCE_TESTWITHOUTNUMPY=1 to test without Numpy.")
        return getattr(np, name)


class test_uses_numpy(ContextDecorator):
    def __enter__(self):
        self.testwithoutnp = False
        if "DTAIDISTANCE_TESTWITHOUTNUMPY" in os.environ and os.environ["DTAIDISTANCE_TESTWITHOUTNUMPY"] == "1":
            self.testwithoutnp = True
        return NumpyStub(self.testwithoutnp)

    def __exit__(self, *exc):
        if self.testwithoutnp:
            if exc[0] is None:
                return
            if issubclass(exc[0], NumpyException):
                return True
