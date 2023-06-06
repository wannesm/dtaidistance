{% macro affstep(s1, s2, s3) -%}
            d = exp(-gamma * d);
            dtw_prev = MAX3(wps[ri_width  + wpsi {{s1}}] - p.penalty,
                            wps[ri_widthp + wpsi {{s2}}], // diagonal
                            wps[ri_widthp + wpsi {{s3}}] - p.penalty);
            if (d < tau) {
                dtw_prev = delta + delta_factor * dtw_prev;
            } else {
                dtw_prev = d + dtw_prev;
            }
            if (dtw_prev < 0) {
                dtw_prev = 0;
            }
            wps[ri_width + wpsi] = dtw_prev;
{%- endmacro -%}
{%- if "affinity" in suffix -%}
{%- set infinity="-INFINITY" -%}
{%- else -%}
{%- set infinity="INFINITY" -%}
{%- endif -%}


{%- if "euclidean" == inner_dist %}
{%- set suffix2="_euclidean" %}
{%- else %}
{%- set suffix2="" %}
{%- endif %}
seq_t dtw_warping_paths{{ suffix }}{{ suffix2 }}(seq_t *wps,
                        seq_t *s1, idx_t l1,
                        seq_t *s2, idx_t l2,
                        bool return_dtw, bool do_sqrt, bool psi_neg,
                        {% if "affinity" in suffix -%}
                        bool only_triu,
                        {% endif -%}
                        {% if "ndim" in suffix -%}
                        int ndim,
                        {% endif -%}
                        {% if "affinity" in suffix -%}
                        seq_t gamma, seq_t tau, seq_t delta, seq_t delta_factor,
                        {% endif -%}
                        DTWSettings *settings) {

    {%- if inner_dist != "euclidean" %}
    if (settings->inner_dist == 1) {
        return dtw_warping_paths{{ suffix }}_euclidean(wps, s1, l1, s2, l2,
                return_dtw, do_sqrt, psi_neg, {% if "affinity" in suffix %}only_triu,{% endif %}{% if "ndim" in suffix %}ndim, {% endif %}{% if "affinity" in suffix -%}gamma, tau, delta, delta_factor, {% endif %}settings);
    }
    {%- endif %}
    {%- if "affinity" in suffix %}
    seq_t dtw_prev;
    {%- else %}
    // DTWPruned
    idx_t sc = 0;
    idx_t ec = 0;
    idx_t ec_next;
    bool smaller_found;
    {%- endif %}

    DTWWps p = dtw_wps_parts(l1, l2, settings);

    {%- if "affinity" not in suffix %}
    if (settings->use_pruning || settings->only_ub) {
        if (ndim == 1) {
            p.max_dist = ub_euclidean(s1, l1, s2, l2);
        } else {
            p.max_dist = ub_euclidean_ndim(s1, l1, s2, l2, ndim);
        }
        {%- if "euclidean" == inner_dist %}
        {%- else %}
        p.max_dist = pow(p.max_dist, 2);
        {%- endif %}
        if (settings->only_ub) {
            if (do_sqrt) {
                return sqrt(p.max_dist);
            } else {
                return p.max_dist;
            }
        }
    }
    {%- endif %}

    idx_t ri, ci, min_ci, max_ci, wpsi, wpsi_start;

    // Top row: ri = -1
    for (wpsi=0; wpsi<settings->psi_2b+1; wpsi++) {
        // ci = wpsi - 1
        wps[wpsi] = 0;
    }
    for (wpsi=settings->psi_2b+1; wpsi<p.width; wpsi++) {
        // MIN(window+ldiffc-1,l2) would be enough
        // ci = wpsi - 1
        wps[wpsi] = {{infinity}};
    }
    // First column:
    wpsi = p.width;
    for (ri=0; ri<settings->psi_1b; ri++) {
        wps[wpsi] = 0;
        wpsi += p.width;
    }
    for (; ri<l1; ri++) {
        wps[wpsi] = {{infinity}};
        wpsi += p.width;
    }

    // dtw_print_wps_compact(wps, l1, l2, settings);
    // dtw_print_wps(wps, l1, l2, settings);
    idx_t ri_widthp = 0;       // ri*width = 0*width = 0
    idx_t ri_width = p.width;  // (ri+1)*width = (0+1)*width = width
    seq_t d;
    idx_t ri_idx, ci_idx;

    // This function splits the loop in four parts that result in different branches
    // that would otherwise be in the loop (and are deterministic).

    // A. Rows: 0 <= ri < min(overlap_left_ri, overlap_right_ri)
    // [0 0 x x x]
    // [0 0 0 x x]
    min_ci = 0;
    max_ci = p.window + p.ldiffc; // ri < overlap_right_i
    for (ri=0; ri<p.ri1; ri++) {
        ri_idx = ri * ndim;
        ci = min_ci;
        wpsi = 1; // index for min_ci
        {%- if "affinity" in suffix %}
        if (only_triu) {
            if (ci < ri) {
                for (; ci<ri; ci++) {
                    wps[ri_width + wpsi] = -INFINITY;
                    wpsi++;
                }
            }
        }
        {%- else %}
        // PrunedDTW
        if (sc <= min_ci) {} else {
            for (; ci<sc; ci++) {
                wps[ri_width + wpsi] = {{infinity}};
                wpsi++;
            }
        }
        smaller_found = false;
        ec_next = ri;
        {%- endif %}
        // A region assumes wps has the same column indices in the previous row
        for (; ci<max_ci; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += SEDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            {%- if "euclidean" == inner_dist %}
            d = sqrt(d);
            {%- endif %}
            {%- if "affinity" in suffix %}
            {{ affstep("-1", "-1", "  ") }}
            {%- else %}
            if (d > p.max_step) { wps[ri_width + wpsi] = {{infinity}}; wpsi++; continue;}
            wps[ri_width + wpsi] = d + MIN3(wps[ri_width  + wpsi - 1] + p.penalty,
                                            wps[ri_widthp + wpsi - 1], // diagonal
                                            wps[ri_widthp + wpsi] + p.penalty);
            // PrunedDTW
            if (wps[ri_width + wpsi] <= p.max_dist) {
                smaller_found = true;
                ec_next = ci + 1;
            } else {
                if (!smaller_found)
                    sc = ci + 1;
                if (ci >= ec)
                    break;
            }
            {%- endif %}
            wpsi++;
        }
        {%- if "affinity" not in suffix %}
        ec = ec_next;
        {%- endif %}
        for (idx_t i=ri_width + wpsi; i<ri_width + p.width; i++) {
            wps[i] = {{infinity}};
        }
        max_ci++;
        ri_widthp = ri_width;
        ri_width += p.width;
    }

    // B. Rows: min(overlap_left_ri, overlap_right_ri) <= ri < overlap_left_ri
    // [0 0 0 0 0]
    // [0 0 0 0 0]
    min_ci = 0;
    max_ci = l2; // ri >= overlap_right_i
    for (ri=p.ri1; ri<p.ri2; ri++) {
        ri_idx = ri * ndim;
        wpsi = 1;
        ci = min_ci;
        {%- if "affinity" in suffix %}
        if (only_triu) {
            if (ci < ri) {
                for (; ci<ri; ci++) {
                    wps[ri_width + wpsi] = -INFINITY;
                    wpsi++;
                }
            }
        }
        {%- else %}
        // PrunedDTW
        if (sc <= min_ci) {} else {
            for (; ci<sc; ci++) {
                wps[ri_width + wpsi] = {{infinity}};
                wpsi++;
            }
        }
        smaller_found = false;
        ec_next = ri;
        {%- endif %}
        for (; ci<max_ci; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += SEDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            {%- if "euclidean" == inner_dist %}
            d = sqrt(d);
            {%- endif %}
            {%- if "affinity" in suffix %}
            {{ affstep("-1", "-1", "  ") }}
            {%- else %}
            if (d > p.max_step) { wps[ri_width + wpsi] = {{infinity}}; wpsi++; continue;}
            // B-region assumes wps has the same column indices in the previous row
            wps[ri_width + wpsi] = d + MIN3(wps[ri_width  + wpsi - 1] + p.penalty,
                                            wps[ri_widthp + wpsi - 1],  // Diagonal
                                            wps[ri_widthp + wpsi] + p.penalty);
            // PrunedDTW
            if (wps[ri_width + wpsi] <= p.max_dist) {
                smaller_found = true;
                ec_next = ci + 1;
            } else {
                if (!smaller_found)
                    sc = ci + 1;
                if (ci >= ec)
                    break;
            }
            {%- endif %}
            wpsi++;
        }
        {%- if "affinity" not in suffix %}
        ec = ec_next;
        {%- endif %}
        for (idx_t i=ri_width + wpsi; i<ri_width + p.width; i++) {
            wps[i] = {{infinity}};
        }
        ri_widthp = ri_width;
        ri_width += p.width;
    }

    // C. Rows: overlap_left_ri <= ri < MAX(parts.overlap_left_ri, parts.overlap_right_ri)
    // [x 0 0 x x]
    // [x x 0 0 x]
    min_ci = 1;
    max_ci = 1 + 2 * p.window - 1 + p.ldiff;
    for (ri=p.ri2; ri<p.ri3; ri++) {
        ri_idx = ri * ndim;
        ci = min_ci;
        wps[ri_width] = {{infinity}};
        wpsi = 1;
        {%- if "affinity" in suffix %}
        if (only_triu) {
            if (ci < ri) {
                for (; ci<ri; ci++) {
                    wps[ri_width + wpsi] = -INFINITY;
                    wpsi++;
                }
            }
        }
        {%- else %}
        // PrunedDTW
        if (sc <= min_ci) {} else {
            for (; ci<sc; ci++) {
                wps[ri_width + wpsi] = {{infinity}};
                wpsi++;
            }
        }
        smaller_found = false;
        ec_next = ri;
        {%- endif %}
        for (; ci<max_ci; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += SEDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            {%- if "euclidean" == inner_dist %}
            d = sqrt(d);
            {%- endif %}
            {%- if "affinity" in suffix %}
            {{ affstep("-1", "  ", "+1") }}
            {%- else %}
            if (d > p.max_step) { wps[ri_width + wpsi] = {{infinity}}; wpsi++; continue;}
            // C-region assumes wps has the column indices in the previous row shifted by one
            wps[ri_width + wpsi] = d + MIN3(wps[ri_width  + wpsi - 1] + p.penalty,
                                            wps[ri_widthp + wpsi],  // Diagonal
                                            wps[ri_widthp + wpsi + 1] + p.penalty);
            // PrunedDTW
            if (wps[ri_width + wpsi] <= p.max_dist) {
                smaller_found = true;
                ec_next = ci + 1;
            } else {
                if (!smaller_found)
                    sc = ci + 1;
                if (ci >= ec)
                    break;
            }
            {%- endif %}
            wpsi++;
        }
        {%- if "affinity" not in suffix %}
        ec = ec_next;
        {%- endif %}
        for (idx_t i=ri_width + wpsi; i<ri_width + p.width; i++) {
            wps[i] = {{infinity}};
        }
        min_ci++;
        max_ci++;
        ri_widthp = ri_width;
        ri_width += p.width;
    }

    // D. Rows: MAX(overlap_left_ri, overlap_right_ri) < ri <= l1
    // [x 0 0 0 0]
    // [x x 0 0 0]
    min_ci = MAX(0, p.ri3 + 1 - p.window - p.ldiff );
    wpsi_start = 2;
    if (p.ri2 == p.ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
    } else {
        min_ci = 1 + p.ri3 - p.ri2;
    }
    for (ri=p.ri3; ri<l1; ri++) {
        ri_idx = ri * ndim;
        ci = min_ci;
        wpsi = wpsi_start;
        for (idx_t i=ri_width; i<(ri_width + wpsi); i++) {
            wps[i] = {{infinity}};
        }
        {%- if "affinity" in suffix %}
        if (only_triu) {
            if (ci < ri) {
                for (; ci<ri; ci++) {
                    wps[ri_width + wpsi] = -INFINITY;
                    wpsi++;
                }
            }
        }
        {%- else %}
        // PrunedDTW
        if (sc <= min_ci) {} else {
            for (; ci<sc; ci++) {
                wps[ri_width + wpsi] = {{infinity}};
                wpsi++;
            }
        }
        smaller_found = false;
        ec_next = ri;
        {%- endif %}
        for (; ci<l2; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += SEDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            {%- if "euclidean" == inner_dist %}
            d = sqrt(d);
            {%- endif %}
            {%- if "affinity" in suffix %}
            {{ affstep("-1", "-1", "  ") }}
            {%- else %}
            if (d > p.max_step) { wps[ri_width + wpsi] = {{infinity}}; wpsi++; continue;}
            // D-region assumes wps has the same column indices in the previous row
            wps[ri_width + wpsi] = d + MIN3(wps[ri_width  + wpsi - 1] + p.penalty,
                                            wps[ri_widthp + wpsi - 1],  // Diagonal
                                            wps[ri_widthp + wpsi] + p.penalty);
            // PrunedDTW
            if (wps[ri_width + wpsi] <= p.max_dist) {
                smaller_found = true;
                ec_next = ci + 1;
            } else {
                if (!smaller_found)
                    sc = ci + 1;
                if (ci >= ec)
                    break;
            }
            {%- endif %}
            wpsi++;
        }
        {%- if "affinity" not in suffix %}
        ec = ec_next;
        {%- endif %}
        for (idx_t i=ri_width + wpsi; i<ri_width + p.width; i++) {
            wps[i] = {{infinity}};
        }
        // printf("%zi [", ri);
        // for (idx_t i=ri_width; i<ri_width + p.width; i++) {
        //     printf("%7.3f, ", wps[i]);
        // }
        // printf("]\n");
        wpsi_start++;
        min_ci++;
        ri_widthp = ri_width;
        ri_width += p.width;
    }

//    dtw_print_wps_compact(wps, l1, l2, settings);
//    dtw_print_wps(wps, l1, l2, settings);

    seq_t rvalue = 0;
    idx_t final_wpsi = ri_widthp + wpsi - 1;
    // Deal with Psi-relaxation
    if (return_dtw && settings->psi_1e == 0 && settings->psi_2e == 0) {
        rvalue = wps[final_wpsi];
    } else if (return_dtw) {
        seq_t mir_value = {{infinity}};
        idx_t mir_rel = 0;
        seq_t mic_value = {{infinity}};
        idx_t mic = 0;
        // Find smallest value in last column
        if (settings->psi_1e != 0) {
            wpsi = final_wpsi;
            for (ri=l1-1; ri>l1-settings->psi_1e-2; ri--) {
                if (wps[wpsi] < mir_value) {
                    mir_value = wps[wpsi];
                    mir_rel = ri + 1;
                } else {
                    // pass
                }
                wpsi -= p.width;
            }
        }
        // Find smallest value in last row
        if (settings->psi_2e != 0) {
            wpsi = final_wpsi;
            for (ci=l2-1; ci>l2-settings->psi_2e-2; ci--) {
                if (wps[wpsi] < mic_value) {
                    mic_value = wps[wpsi];
                    mic = ci + 1;
                } else {
                    // pass
                }
                wpsi -= 1;
            }
        }
        // Set values with higher indices than the smallest value to -1
        // and return smallest value as DTW
        if (mir_value < mic_value) {
            // last column has smallest value
            if (psi_neg) {
                for (idx_t ri=mir_rel + 1; ri<l1 + 1; ri++) {
                    wpsi = ri*p.width + (p.width - 1);
                    wps[wpsi] = -1;
                }
            }
            rvalue = mir_value;
        } else {
            // last row has smallest value
            if (psi_neg) {
                for (ci=p.width - (l2 - mic); ci<p.width; ci++) {
                    wpsi = l1*p.width + ci;
                    if (p.window != 0 && p.window != l2) {
                        wpsi--;
                    }
                    wps[wpsi] = -1;
                }
            }
            rvalue =  mic_value;
        }
    } else {
        rvalue = -1;
    }

    if (settings->max_dist > 0 && rvalue > settings->max_dist) {
        // DTWPruned keeps the last value larger than max_dist. Correct for this.
        rvalue = {{infinity}};
    }


    {%- if "euclidean" == inner_dist %}
    {%- else %}
    if (do_sqrt) {
        for (idx_t i=0; i<p.length ; i++) {
            if (wps[i] > 0) {
                wps[i] = sqrt(wps[i]);
            }
        }
        if (return_dtw) {
            if (rvalue > 0) {
                rvalue = sqrt(rvalue);
            }
        }
    }
    {%- endif %}

    return rvalue;
}
