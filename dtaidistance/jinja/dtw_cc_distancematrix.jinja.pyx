{%- set s = " " %}
def distance_matrix{{suffix}}(cur,{{s}}
        {%- if "ndim" in suffix -%}
        int ndim,{{s}}
        {%- endif -%}
        block=None, **kwargs):
    """Compute a distance matrix between all sequences given in `cur`.
    This method calls a pure c implementation of the dtw computation that
    avoids the GIL.

    Assumes C-contiguous arrays.

    :param cur: DTWSeriesMatrix or DTWSeriesPointers
    :param block: see DTWBlock
    :param kwargs: Settings (see DTWSettings)
    :return: The distance matrix as a list representing the triangular matrix.
    """
    cdef DTWSeriesMatrix matrix
    {%- if "ndim" in suffix %}
    cdef DTWSeriesMatrixNDim matrixnd
    {%- endif %}
    cdef DTWSeriesPointers ptrs
    cdef Py_ssize_t length = 0
    cdef Py_ssize_t block_rb=0
    cdef Py_ssize_t block_re=0
    cdef Py_ssize_t block_cb=0
    cdef Py_ssize_t block_ce=0
    cdef Py_ssize_t ri = 0
    if block is not None and block != 0.0:
        block_rb = block[0][0]
        block_re = block[0][1]
        block_cb = block[1][0]
        block_ce = block[1][1]

    settings = DTWSettings(**kwargs)
    cdef DTWBlock dtwblock = DTWBlock(rb=block_rb, re=block_re, cb=block_cb, ce=block_ce)
    if block is not None and block != 0.0 and len(block) > 2 and block[2] is False:
        dtwblock.triu_set(False)
    length = distance_matrix_length(dtwblock, len(cur))

    # Correct block
    if dtwblock.re == 0:
        dtwblock.re_set(len(cur))
    if dtwblock.ce == 0:
        dtwblock.ce_set(len(cur))

    cdef array.array dists = array.array('d')
    array.resize(dists, length)

    if isinstance(cur, DTWSeriesMatrix){{s}}
        {%- if "ndim" in suffix %}or isinstance(cur, DTWSeriesMatrixNDim) {% endif -%}
        or isinstance(cur, DTWSeriesPointers):
        pass
    elif cur.__class__.__name__ == "SeriesContainer":
        cur = cur.c_data_compat()
    else:
        cur = dtw_series_from_data(cur{%- if "ndim" in suffix %}, force_pointers=True{% endif -%})

    if isinstance(cur, DTWSeriesPointers):
        ptrs = cur
        {%- if "ndim" in suffix %}
        dtaidistancec_dtw.dtw_distances_ndim_ptrs(
            ptrs._ptrs, ptrs._nb_ptrs, ptrs._lengths, ndim,
            dists.data.as_{{seq_t}}s, &dtwblock._block, &settings._settings)
        {%- else %}
        dtaidistancec_dtw.dtw_distances_ptrs(
            ptrs._ptrs, ptrs._nb_ptrs, ptrs._lengths,
            dists.data.as_{{seq_t}}s, &dtwblock._block, &settings._settings)
        {%- endif %}
    elif isinstance(cur, DTWSeriesMatrix):
        {%- if "ndim" in suffix %}
        # This is not a n-dimensional case ?
        {%- endif %}
        matrix = cur
        dtaidistancec_dtw.dtw_distances_matrix(
            &matrix._data[0,0], matrix.nb_rows, matrix.nb_cols,
            dists.data.as_{{seq_t}}s, &dtwblock._block, &settings._settings)
    {%- if "ndim" in suffix %}
    elif isinstance(cur, DTWSeriesMatrixNDim):
        matrixnd = cur
        dtaidistancec_dtw.dtw_distances_ndim_matrix(
            &matrixnd._data[0,0,0], matrixnd.nb_rows, matrixnd.nb_cols, ndim,
            dists.data.as_{{seq_t}}s, &dtwblock._block, &settings._settings)
    {%- endif %}
    else:
        raise Exception("Unknown series container")

    return dists
