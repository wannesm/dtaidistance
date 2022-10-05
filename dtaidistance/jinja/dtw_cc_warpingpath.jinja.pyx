{%- set s = " " %}
def warping_path{{suffix}}(
        {%- if "ndim" in suffix -%}
        double[:, :] s1, double[:, :] s2, int ndim=1,{{s}}
        {%- else -%}
        double[:] s1, double[:] s2,{{s}}
        {%- endif -%}
        **kwargs):
    # Assumes C contiguous
    cdef Py_ssize_t path_length;
    settings = DTWSettings(**kwargs)
    cdef Py_ssize_t *i1 = <Py_ssize_t *> PyMem_Malloc((len(s1) + len(s2)) * sizeof(Py_ssize_t))
    if not i1:
        raise MemoryError()
    cdef Py_ssize_t *i2 = <Py_ssize_t *> PyMem_Malloc((len(s1) + len(s2)) * sizeof(Py_ssize_t))
    if not i2:
        raise MemoryError()
    try:
        {%- if "ndim" in suffix %}
        path_length = dtaidistancec_dtw.warping_path_ndim(&s1[0, 0], len(s1), &s2[0, 0], len(s2), i1, i2, ndim, &settings._settings)
        {%- else %}
        path_length = dtaidistancec_dtw.warping_path(&s1[0], len(s1), &s2[0], len(s2), i1, i2, &settings._settings)
        {%- endif %}
        path = []
        for i in range(path_length):
            path.append((i1[i], i2[i]))
        path.reverse()
    finally:
        PyMem_Free(i1)
        PyMem_Free(i2)
    return path
