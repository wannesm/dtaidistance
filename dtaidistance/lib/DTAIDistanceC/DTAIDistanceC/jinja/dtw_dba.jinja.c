/*!
 Barycenter.
 
 Based on:
 F. Petitjean, A. Ketterlin, and P. Gan ̧carski.
 A global averaging method for dynamic time warping, with applications to clustering.
 Pattern Recognition, 44(3):678–693, 2011.

{% if "ptrs" in suffix -%}
 @param ptrs Pointers to arrays.  The arrays are expected to be 1-dimensional.
 @param nb_ptrs Length of ptrs array
 @param lengths Array of length nb_ptrs with all lengths of the arrays in ptrs.
{%- elif "matrix" in suffix -%}
 @param matrix Sequences ordered in a matrix
 @param nb_rows Number of rows
 @param nb_cols Number of columns
{%- endif %}
 @param c Initial average, afterwards the updated average
 @param t Length of average (typically this is the same as nb_cols)
            Real length is t*ndim.
 @param mask Bit-array
 @param prob_samples Probabilistically sample the best path, samples number of times.
        Uses deterministic best path if samples is 0.
 @param settings Settings for distance functions
 */
{%- if "ptrs" in suffix -%}
{%- set nb_size = "nb_ptrs" %}
{%- set max_length = "max_length" %}
{%- set cur_length = "lengths[r]" %}
{%- elif "matrix" in suffix -%}
{%- set nb_size = "nb_rows" %}
{%- set max_length = "nb_cols" %}
{%- set cur_length = "nb_cols" %}
{%- endif %}
void dtw_dba_{{ suffix }}(
                  {%- if "ptrs" in suffix -%}
                  seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths,
                  {%- elif "matrix" in suffix -%}
                  seq_t *matrix, idx_t nb_rows, idx_t nb_cols,
                  {%- endif %}
                  seq_t *c, idx_t t, ba_t *mask, int prob_samples, int ndim,
                  DTWSettings *settings) {
    seq_t *assoctab = (seq_t *)malloc(t * ndim * sizeof(seq_t));
    idx_t *assoctab_cnt = (idx_t *)malloc(t * sizeof(idx_t));
    idx_t r_idx = 0;
    {%- if "ptrs" in suffix %}
    idx_t max_length = 0;
    for (r_idx=0; r_idx<nb_ptrs; r_idx++) {
        if (lengths[r_idx] > max_length) {
            max_length = lengths[r_idx];
        }
    }
    {%- endif %}
    idx_t *ci = (idx_t *)malloc(({{max_length}} + t) * sizeof(idx_t));
    idx_t *mi = (idx_t *)malloc(({{max_length}} + t) * sizeof(idx_t));
    idx_t pi, di;
    seq_t *sequence;
    seq_t *wps;
    seq_t avg_step;
    idx_t path_length;

    idx_t wps_length = dtw_settings_wps_length(t, {{max_length}}, settings);
    wps = (seq_t *)malloc(wps_length * sizeof(seq_t));

    for (pi=0; pi<t; pi++) {
        for (di=0; di<ndim; di++) {
            assoctab[pi * ndim + di] = 0;
        }
        assoctab_cnt[pi] = 0;
    }
    if (prob_samples == 0) {
        for (idx_t r=0; r<{{nb_size}}; r++) {
            {%- if "ptrs" in suffix %}
            sequence = ptrs[r];
            {%- elif "matrix" in suffix %}
            sequence = &matrix[r_idx];
            {%- endif %}
            if (bit_test(mask, r)) {
                // warping_path(c, t, sequence, lengths[r], ci, mi, settings);
                dtw_warping_paths_ndim(wps, c, t, sequence, {{cur_length}}, false, false, true, ndim, settings);
                path_length = dtw_best_path(wps, ci, mi, t, {{cur_length}}, settings);
                // printf("best_path(%zu/%zu) = [", r+1, nb_rows);
                // for (idx_t i=0; i<path_length; i++) {
                //     printf(" %zu:(%zu,%zu)", i, ci[i], mi[i]);
                // }
                // printf("]\n");
                for (pi=0; pi<path_length; pi++) {
                    for (di=0; di<ndim; di++) {
                        assoctab[ci[pi]*ndim+di] += sequence[mi[pi]*ndim+di];
                    }
                    assoctab_cnt[ci[pi]] += 1;
                    // printf("[%zu] = [%zu] += %f\n", ci[pi], mi[pi], sequence[mi[pi]]);
                }
            }
            {%- if "matrix" in suffix %}
            r_idx += nb_cols*ndim;
            {%- endif %}
        }
    } else {
        for (idx_t r=0; r<{{nb_size}}; r++) {
            {%- if "ptrs" in suffix %}
            sequence = ptrs[r];
            {%- elif "matrix" in suffix %}
            sequence = &matrix[r_idx];
            {%- endif %}
            if (bit_test(mask, r)) {
                avg_step = dtw_warping_paths_ndim(wps, c, t, sequence, {{cur_length}}, true, false, true, ndim, settings);
                avg_step /= t;
                for (idx_t i_sample=0; i_sample<prob_samples; i_sample++) {
                    path_length = dtw_best_path_prob(wps, ci, mi, t, {{cur_length}}, avg_step, settings);
                    // printf("best_path_prob = [");
                    // for (idx_t i=0; i<path_length; i++) {
                    //     printf("(%zu,%zu)", ci[i], mi[i]);
                    // }
                    // printf("]\n");
                    for (pi=0; pi<path_length; pi++) {
                        for (di=0; di<ndim; di++) {
                            assoctab[ci[pi]*ndim+di] += sequence[mi[pi]*ndim+di];
                        }
                        assoctab_cnt[ci[pi]] += 1;
                    }
                }
            }
            {%- if "matrix" in suffix %}
            r_idx += nb_cols*ndim;
            {%- endif %}
        }
    }
    for (idx_t i=0; i<t; i++) {
        if (assoctab_cnt[i] != 0) {
            for (di=0; di<ndim; di++) {
                c[i*ndim+di] = assoctab[i*ndim+di] / assoctab_cnt[i];
                // printf("c[%zu] = %f = %f / %zd\n", i, c[i], assoctab[i], assoctab_cnt[i]);
            }
        } else {
            printf("WARNING: assoctab_cnt[%zu] == 0\n", i);
            for (di=0; di<ndim; di++) {
                c[i*ndim+di] = 0;
            }
        }
    }
    free(assoctab);
    free(assoctab_cnt);
    free(ci);
    free(mi);
    free(wps);
}

