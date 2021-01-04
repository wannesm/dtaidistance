

cdef extern from "dd_globals.h":
    ctypedef double seq_t

    void set_srand(int seed);
