/*!
{%- if suffix == "matrix" %}
Distance matrix for n-dimensional DTW, executed on a 2-dimensional array and in parallel.
{%- elif suffix == "ndim_matrix" %}
Distance matrix for n-dimensional DTW, executed on a 3-dimensional array and in parallel.
{%- elif "ptrs" in suffix %}
Distance matrix for n-dimensional DTW, executed on a list of pointers to arrays and in parallel.
{%- endif %}

@see dtw_distances_{{suffix}}
*/
{%- if "ptrs" in suffix %}
{%- set nb_size_r = "nb_ptrs" %}
{%- set nb_size_c = "nb_ptrs" %}
{%- elif "matrix" in suffix %}
{%- set nb_size_r = "nb_rows" %}
{%- set nb_size_c = "nb_rows" %}
{%- elif "matrices" in suffix %}
{%- set nb_size_r = "nb_rows_r" %}
{%- set nb_size_c = "nb_rows_c" %}
{%- endif %}
idx_t dtw_distances_{{suffix}}_parallel(
        {%- if suffix == "ptrs" -%}
        seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths,
        {%- elif suffix == "ndim_ptrs" -%}
        seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths, int ndim,
        {%- elif suffix == "matrix" -%}
        seq_t *matrix, idx_t nb_rows, idx_t nb_cols,
        {%- elif suffix == "ndim_matrix" -%}
        seq_t *matrix, idx_t nb_rows, idx_t nb_cols, int ndim,
        {%- elif suffix == "matrices" -%}
        seq_t *matrix_r, idx_t nb_rows_r, idx_t nb_cols_r,
                          seq_t *matrix_c, idx_t nb_rows_c, idx_t nb_cols_c,
        {%- elif suffix == "ndim_matrices" -%}
        seq_t *matrix_r, idx_t nb_rows_r, idx_t nb_cols_r,
                          seq_t *matrix_c, idx_t nb_rows_c, idx_t nb_cols_c, int ndim,
        {%- endif %}
                          seq_t* output, DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, r_i, c_i;
    idx_t length;
    idx_t *cbs, *rls;

    if (dtw_distances_prepare(block, {{nb_size_r}}, {{nb_size_c}}, &cbs, &rls, &length, settings) != 0) {
        return 0;
    }
    
#if defined(_OPENMP)
    r_i=0;
    {%- if suffix == "ptrs" %}
    // Rows have different lengths, thus use guided scheduling to make threads with shorter rows
    // not wait for threads with longer rows. Also the first rows are always longer than the last
    // ones (upper triangular matrix), so this nicely aligns with the guided strategy.
    // Using schedule("static, 1") is also fast for the same reason (neighbor rows are almost
    // the same length, thus a circular assignment works well) but assumes all DTW computations take
    // the same amount of time.
    {%- endif %}
    #pragma omp parallel for private(r_i, c_i, r, c) schedule(guided)
    for (r_i=0; r_i < (block->re - block->rb); r_i++) {
        r = block->rb + r_i;
        c_i = 0;
        if (block->triu) {
            c = cbs[r_i];
        } else {
            c = block->cb;
        }
        for (; c<block->ce; c++) {
            {%- if suffix == "ptrs" %}
            double value = dtw_distance(ptrs[r], lengths[r],
                                        ptrs[c], lengths[c], settings);
            {%- elif suffix == "ndim_ptrs" %}
            double value = dtw_distance_ndim(ptrs[r], lengths[r],
                          ptrs[c], lengths[c],
                          ndim, settings);
            {%- elif suffix == "matrix" %}
            double value = dtw_distance(&matrix[r*nb_cols], nb_cols,
                                         &matrix[c*nb_cols], nb_cols, settings);
            {%- elif suffix == "ndim_matrix" %}
            double value = dtw_distance_ndim(&matrix[r*nb_cols*ndim], nb_cols,
                                             &matrix[c*nb_cols*ndim], nb_cols,
                                             ndim, settings);
            {%- elif suffix == "matrices" %}
            double value = dtw_distance(&matrix_r[r*nb_cols_r], nb_cols_r,
                                        &matrix_c[c*nb_cols_c], nb_cols_c, settings);
            {%- elif suffix == "ndim_matrices" %}
            double value = dtw_distance_ndim(&matrix_r[r*nb_cols_r*ndim], nb_cols_r,
                                             &matrix_c[c*nb_cols_c*ndim], nb_cols_c,
                                             ndim, settings);
            {%- endif %}
            if (block->triu) {
                output[rls[r_i] + c_i] = value;
            } else {
                output[(block->ce - block->cb) * r_i + c_i] = value;
            }
            c_i++;
        }
    }
    
    if (block->triu) {
        free(cbs);
        free(rls);
    }
    return length;
#else
    printf("ERROR: DTAIDistanceC is compiled without OpenMP support.\n");
    for  (r_i=0; r_i<length; r_i++) {
        output[r_i] = 0;
    }
    return 0;
#endif
}

