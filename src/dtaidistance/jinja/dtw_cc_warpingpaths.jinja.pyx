{% macro select_c_fn(wps_view) -%}
    {%- if "affinity" in suffix %}
    {%- if "ndim" in suffix %}
    d = dtaidistancec_dtw.dtw_warping_paths_affinity_ndim(&{{wps_view}}[0,0], &s1[0,0], len(s1), &s2[0,0], len(s2),
                                                          True, True, psi_neg, only_triu, ndim,
                                                          gamma, tau, delta, delta_factor,
                                                          &settings._settings)
    {%- else %}
    d = dtaidistancec_dtw.dtw_warping_paths_affinity(&{{wps_view}}[0,0], &s1[0], len(s1), &s2[0], len(s2),
                                                     True, True, psi_neg, only_triu,
                                                     gamma, tau, delta, delta_factor,
                                                     &settings._settings)
    {%- endif %}
    {%- else %}
    {%- if "ndim" in suffix %}
    d = dtaidistancec_dtw.dtw_warping_paths_ndim(&{{wps_view}}[0,0], &s1[0,0], len(s1), &s2[0,0], len(s2),
                                                 True, keep_int_repr, psi_neg, ndim, &settings._settings)
    {%- else %}
    d = dtaidistancec_dtw.dtw_warping_paths(&{{wps_view}}[0,0], &s1[0], len(s1), &s2[0], len(s2),
                                            True, keep_int_repr, psi_neg, &settings._settings)
    {%- endif %}
    {%- endif %}
{%- endmacro -%}
{%- set s = " " -%}


def warping_paths{{ suffix }}(
        {%- if "ndim" in suffix -%}
        seq_t[:, :] dtw, seq_t[:, :] s1, seq_t[:, :] s2,{{s}}
        {%- else -%}
        seq_t[:, :] dtw, seq_t[:] s1, seq_t[:] s2,{{s}}
        {%- endif -%}
        {%- if "affinity" in suffix %}
        bint only_triu, seq_t gamma, seq_t tau, seq_t delta, seq_t delta_factor,
        {%- endif %}
        bint psi_neg=False, bint keep_int_repr=False, **kwargs):
    {%- if "ndim" in suffix %}
    ndim = s1.shape[1]
    if s1.shape[1] != s2.shape[1]:
        raise Exception("Dimension of sequence entries needs to be the same: {} != {}".format(s1.shape[1], s2.shape[1]))
    {%- endif %}
    # Assumes C contiguous
    settings = DTWSettings(**kwargs)
    dtw_length = dtw.shape[0] * dtw.shape[1]
    req_length = dtaidistancec_dtw.dtw_settings_wps_length(len(s1), len(s2), &settings._settings)
    req_width = dtaidistancec_dtw.dtw_settings_wps_width(len(s1), len(s2), &settings._settings)
    shape = (1, req_length)
    if req_length == dtw_length and req_width == dtw.shape[1]:
        # No compact WPS array is required
        wps = dtw
    else:
        try:
            # Use cython.view.array to avoid numpy dependency
            wps = cvarray(shape=shape, itemsize=sizeof(seq_t), format="{{seq_format}}")
        except MemoryError as exc:
            print("Cannot allocate memory for warping paths matrix. Trying " + str(shape) + ".")
            raise exc
    cdef seq_t [:, :] wps_view = wps
    cdef seq_t d
    {{ select_c_fn("wps_view")}}
    if not (req_length == dtw_length and req_width == dtw.shape[1]):
        {%- if "affinity" in suffix %}
        dtaidistancec_dtw.dtw_expand_wps_affinity(&wps_view[0,0], &dtw[0, 0], len(s1), len(s2), &settings._settings)
        {%- else %}
        dtaidistancec_dtw.dtw_expand_wps(&wps_view[0,0], &dtw[0, 0], len(s1), len(s2), &settings._settings)
        {%- endif %}
    return d


def warping_paths_compact{{ suffix }}(
        {%- if "ndim" in suffix -%}
        seq_t[:, :] dtw, seq_t[:, :] s1, seq_t[:, :] s2,{{s}}
        {%- else -%}
        seq_t[:, :] dtw, seq_t[:] s1, seq_t[:] s2,{{s}}
        {%- endif -%}
        {%- if "affinity" in suffix %}
        bint only_triu, seq_t gamma, seq_t tau, seq_t delta, seq_t delta_factor,
        {%- endif %}
        bint psi_neg=False, bint keep_int_repr=False, **kwargs):
    {%- if "ndim" in suffix %}
    if s1.shape[1] != s2.shape[1]:
        raise Exception("Dimension of sequence entries needs to be the same: {} != {}".format(s1.shape[1], s2.shape[1]))
    ndim = s1.shape[1]
    {%- endif %}
    # Assumes C contiguous
    settings = DTWSettings(**kwargs)
    cdef seq_t d
    {{ select_c_fn("dtw") }}
    return d
