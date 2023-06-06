{%- set s = " " %}
def warping_path{{suffix}}(
        {%- if "ndim" in suffix -%}
        seq_t[:, :] s1, seq_t[:, :] s2, int ndim=1,{{s}}
        {%- else -%}
        seq_t[:] s1, seq_t[:] s2,{{s}}
        {%- endif -%}
        include_distance=False, **kwargs):
    # Assumes C contiguous
    cdef Py_ssize_t path_length;
    cdef seq_t dist;
    settings = DTWSettings(**kwargs)
    cdef Py_ssize_t *i1 = <Py_ssize_t *> PyMem_Malloc((len(s1) + len(s2)) * sizeof(Py_ssize_t))
    if not i1:
        raise MemoryError()
    cdef Py_ssize_t *i2 = <Py_ssize_t *> PyMem_Malloc((len(s1) + len(s2)) * sizeof(Py_ssize_t))
    if not i2:
        raise MemoryError()
    try:
        {%- if "ndim" in suffix %}
        dist = dtaidistancec_dtw.dtw_warping_path_ndim(&s1[0, 0], len(s1), &s2[0, 0], len(s2), i1, i2, &path_length, ndim, &settings._settings)
        {%- else %}
        dist = dtaidistancec_dtw.dtw_warping_path(&s1[0], len(s1), &s2[0], len(s2), i1, i2, &path_length, &settings._settings)
        {%- endif %}
        path = []
        for i in range(path_length):
            path.append((i1[i], i2[i]))
        path.reverse()
    finally:
        PyMem_Free(i1)
        PyMem_Free(i2)
    if include_distance:
        return path, dist
    return path
