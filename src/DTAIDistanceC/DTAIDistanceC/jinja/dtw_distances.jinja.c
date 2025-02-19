/*!
{%- if suffix == "ptrs" %}
Distance matrix for n-dimensional DTW, executed on a list of pointers to arrays.
{%- elif suffix == "ndim_ptrs" %}
Distance matrix for n-dimensional DTW, executed on a list of pointers to arrays.
{%- elif suffix == "matrix" %}
Distance matrix for n-dimensional DTW, executed on a 2-dimensional array.
{%- elif suffix == "ndim_matrix" %}
Distance matrix for n-dimensional DTW, executed on a 3-dimensional array.
{%- elif suffix == "matrices" %}
Distance matrix for n-dimensional DTW, executed on two 2-dimensional arrays.
{%- elif suffix == "ndim_matrices" %}
Distance matrix for n-dimensional DTW, executed on two 3-dimensional arrays.
{%- endif %}
{%- if "matrix" in suffix %}

 The array is assumed to be C contiguous: C contiguous means that the array data is continuous in memory (see below) and that neighboring elements in the first dimension of the array are furthest apart in memory, whereas neighboring elements in the last dimension are closest together (from https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#brief-recap-on-c-fortran-and-strided-memory-layouts).
{%- endif %}
{% if suffix == "ptrs" %}
@param ptrs Pointers to arrays.  The arrays are expected to be 1-dimensional.
@param nb_ptrs Length of ptrs array
@param lengths Array of length nb_ptrs with all lengths of the arrays in ptrs.
{%- elif suffix == "ndim_ptrs" %}
@param ptrs Pointers to arrays. The order is defined by 1st dim is sequence entry, 2nd dim are the n-dimensional values. Thus the values for each n-dimensional entry are next to each other in the memory layout of the array.
@param nb_ptrs Length of ptrs array
@param lengths Array of length nb_ptrs with all lengths of the arrays in ptrs.
@param ndim The number of dimensions in each sequence entry
{%- elif suffix == "matrix" %}
@param matrix 2-dimensional array. The order is defined by 1st dimension are the series, the 2nd dimension are the sequence entries.
@param nb_rows Number of series, size of the 1st dimension of matrix
@param nb_cols Number of elements in each series, size of the 2nd dimension of matrix
{%- elif suffix == "ndim_matrix" %}
@param matrix 3-dimensional array. The order is defined by 1st dimension are the series, the 2nd dimension are the sequence entries, and the 3rd dimension are the n-dimensional values.
@param nb_rows Number of series, size of the 1st dimension of matrix
@param nb_cols Number of elements in each series, size of the 2nd dimension of matrix
@param ndim The number of dimensions in each sequence entry, size of the 3rd dimension of matrix
{%- endif %}
@param output Array to store all outputs (should be (nb_ptrs-1)*nb_ptrs/2 if no block is given)
@param block Restrict to a certain block of combinations of series.
@param settings DTW settings
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
idx_t dtw_distances_{{ suffix }}(
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
    idx_t r, c, cb;
    idx_t length;
    idx_t i;
    seq_t value;

    length = dtw_distances_length(block, {{nb_size_r}}, {{nb_size_c}});
    if (length == 0) {
        return 0;
    }

    // Correct block
    if (block->re == 0) {
        block->re = {{ nb_size_r }};
    }
    if (block->ce == 0) {
        block->ce = {{ nb_size_c }};
    }

    i = 0;
    for (r=block->rb; r<block->re; r++) {
        if (block->triu && r + 1 > block->cb) {
            cb = r+1;
        } else {
            cb = block->cb;
        }
        for (c=cb; c<block->ce; c++) {
            {%- if suffix == "ptrs" %}
            value = dtw_distance(ptrs[r], lengths[r],
                                 ptrs[c], lengths[c], settings);
            {%- elif suffix == "ndim_ptrs" %}
            value = dtw_distance_ndim(ptrs[r], lengths[r],
                                      ptrs[c], lengths[c],
                                      ndim, settings);
            {%- elif suffix == "matrix" %}
            value = dtw_distance(&matrix[r*nb_cols], nb_cols,
                                 &matrix[c*nb_cols], nb_cols, settings);
            {%- elif suffix == "ndim_matrix" %}
            value = dtw_distance_ndim(&matrix[r*nb_cols*ndim], nb_cols,
                                      &matrix[c*nb_cols*ndim], nb_cols,
                                      ndim, settings);
            {%- elif suffix == "matrices" %}
            value = dtw_distance(&matrix_r[r*nb_cols_r], nb_cols_r,
                                 &matrix_c[c*nb_cols_c], nb_cols_c, settings);
            {%- elif suffix == "ndim_matrices" %}
            value = dtw_distance_ndim(&matrix_r[r*nb_cols_r*ndim], nb_cols_r,
                                      &matrix_c[c*nb_cols_c*ndim], nb_cols_c,
                                      ndim, settings);
            {%- endif %}
            // printf("i=%zu - r=%zu - c=%zu - value=%.4f\n", i, r, c, value);
            output[i] = value;
            i += 1;
        }
    }
    assert(length == i);
    return length;
}

