

class PackageMissingException(Exception):
    def __init__(self, message):
        super().__init__(message)


class NumpyException(PackageMissingException):
    def __init__(self, message):
        super().__init__(message)


class MatplotlibException(PackageMissingException):
    def __init__(self, message):
        super().__init__(message)


class ScipyException(PackageMissingException):
    def __init__(self, message):
        super().__init__(message)


class CythonException(PackageMissingException):
    def __init__(self, message):
        super().__init__(message)


class PyClusteringException(PackageMissingException):
    def __init__(self, message):
        super().__init__(message)


class PandasException(PackageMissingException):
    def __init__(self, message):
        super().__init__(message)
