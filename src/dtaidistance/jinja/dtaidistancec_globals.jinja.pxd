

cdef extern from "dd_globals.h":
    ctypedef {{seq_t}} seq_t

    ctypedef struct DDPathEntry:
        Py_ssize_t i
        Py_ssize_t j

    ctypedef struct DDPath:
        DDPathEntry * array
        Py_ssize_t length
        Py_ssize_t capacity
        seq_t distance

    void dd_path_free(DDPath *a)


