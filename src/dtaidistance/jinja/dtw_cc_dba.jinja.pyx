{%- set s = " " %}
{%- if "ndim" in suffix %}
{%- set ndim = "ndim" %}
{%- else %}
{%- set ndim = "1" %}
{%- endif %}
def dba{{suffix}}(cur,{{s}}
        {%- if "ndim" in suffix -%}
        seq_t[:, :] c, unsigned char[:] mask, int nb_prob_samples, int ndim,{{s}}
        {%- else -%}
        seq_t[:] c, unsigned char[:] mask, int nb_prob_samples,{{s}}
        {%- endif -%}
        **kwargs):
    {%- if "ndim" in suffix %}
    cdef seq_t *c_ptr = &c[0, 0];
    {%- else %}
    cdef seq_t *c_ptr = &c[0];
    {%- endif %}
    cdef unsigned char *mask_ptr = &mask[0];
    settings = DTWSettings(**kwargs)
    dba_inner(cur, c_ptr, len(c), mask_ptr, nb_prob_samples, {{ndim}}, settings)
    return c
