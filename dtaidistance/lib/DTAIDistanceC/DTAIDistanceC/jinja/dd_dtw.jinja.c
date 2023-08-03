/*!
@file dtw.c
@brief DTAIDistance.dtw

@author Wannes Meert
@copyright Copyright © 2020 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/
#include "dd_dtw.h"


//#define DTWDEBUG


// MARK: Settings

/* Create settings struct with default values (all extras deactivated). */
DTWSettings dtw_settings_default(void) {
    DTWSettings s = {
        .window = 0,
        .max_dist = 0,
        .max_step = 0,
        .max_length_diff = 0,
        .penalty = 0,
        .psi_1b = 0,
        .psi_1e = 0,
        .psi_2b = 0,
        .psi_2e = 0,
        .use_pruning = false,
        .only_ub = false,
        .inner_dist = 0,
        .window_type = 0
    };
    return s;
}

idx_t dtw_settings_wps_length(idx_t l1, idx_t l2, DTWSettings *settings) {
    DTWWps p = dtw_wps_parts(l1, l2, settings);
    return p.length;
}

idx_t dtw_settings_wps_width(idx_t l1, idx_t l2, DTWSettings *settings) {
    DTWWps p = dtw_wps_parts(l1, l2, settings);
    return p.width;
}

void dtw_settings_set_psi(idx_t psi, DTWSettings *settings) {
    settings->psi_1b = psi;
    settings->psi_1e = psi;
    settings->psi_2b = psi;
    settings->psi_2e = psi;
}

void dtw_settings_print(DTWSettings *settings) {
    printf("DTWSettings {\n");
    printf("  window = %zu\n", settings->window);
    printf("  max_dist = %f\n", settings->max_dist);
    printf("  max_step = %f\n", settings->max_step);
    printf("  max_length_diff = %zu\n", settings->max_length_diff);
    printf("  penalty = %f\n", settings->penalty);
    printf("  psi = [%zu, %zu, %zu, %zu]\n", settings->psi_1b, settings->psi_1e,
                                             settings->psi_2b, settings->psi_2e);
    printf("  use_pruning = %d\n", settings->use_pruning);
    printf("  only_ub = %d\n", settings->only_ub);
    printf("  inner_dist = %d\n", settings->inner_dist);
    printf("  window_type = %d\n", settings->window_type);
    printf("}\n");
}

// MARK: DTW

{% set suffix = '' %}
{% set inner_dist = 'squaredeuclidean' %}
{%- include 'dtw_distance.jinja.c' %}

{% set suffix = '_ndim' %}
{% set inner_dist = 'squaredeuclidean' %}
{%- include 'dtw_distance.jinja.c' %}

{% set suffix = '' %}
{% set inner_dist = 'euclidean' %}
{%- include 'dtw_distance.jinja.c' %}

{% set suffix = '_ndim' %}
{% set inner_dist = 'euclidean' %}
{%- include 'dtw_distance.jinja.c' %}

// MARK: WPS

/*!
Compute all warping paths between two series.
 
@param wps Empty array of length `(l1+1)*min(l2+1, abs(l1-l2) + 2*window-1)` in which the warping paths will be stored.
    It represents the full matrix of warping paths between the two series.
@param s1 First sequence
@param l1 Length of first sequence
@param s2 Second sequence
@param l2 Length of second sequence
@param return_dtw If only the matrix is required, finding the dtw value can be skipped
    to save operations.
@param do_sqrt Apply the sqrt operations on all items in the wps array. If not required,
    this can be skipped to save operations.
@param psi_neg For psi-relaxation, replace non-optimal values with -1
@param settings A DTWSettings struct with options for the DTW algorithm.
 
@return The dtw value if return_dtw is true; Otherwise -1.
*/
seq_t dtw_warping_paths(seq_t *wps,
                        seq_t *s1, idx_t l1,
                        seq_t *s2, idx_t l2,
                        bool return_dtw, bool do_sqrt, bool psi_neg,
                        DTWSettings *settings) {
    return dtw_warping_paths_ndim(wps, s1, l1, s2, l2,
                             return_dtw, do_sqrt, psi_neg, 1,
                             settings);
}

{% set suffix = '_ndim' %}
{% set inner_dist = 'squaredeuclidean' %}
{%- include 'dtw_warpingpaths.jinja.c' %}

seq_t dtw_warping_paths_euclidean(
        seq_t *wps,
        seq_t *s1, idx_t l1,
        seq_t *s2, idx_t l2,
        bool return_dtw, bool do_sqrt, bool psi_neg,
        DTWSettings *settings) {
    return dtw_warping_paths_ndim_euclidean(
            wps, s1, l1, s2, l2,
            return_dtw, do_sqrt, psi_neg, 1,
            settings);
}

{% set suffix = '_ndim' %}
{% set inner_dist = 'euclidean' %}
{%- include 'dtw_warpingpaths.jinja.c' %}


{% set suffix = '' %}
{%- include 'dtw_expandwps.jinja.c' %}


seq_t dtw_warping_paths_affinity(seq_t *wps,
                        seq_t *s1, idx_t l1,
                        seq_t *s2, idx_t l2,
                        bool return_dtw, bool do_sqrt, bool psi_neg, bool only_triu,
                        seq_t gamma, seq_t tau, seq_t delta, seq_t delta_factor,
                        DTWSettings *settings) {
    return dtw_warping_paths_affinity_ndim(wps, s1, l1, s2, l2,
                             return_dtw, do_sqrt, psi_neg, only_triu, 1,
                             gamma, tau, delta, delta_factor,
                             settings);
}


{% set suffix = '_affinity_ndim' %}
{% set inner_dist = 'squaredeuclidean' %}
{%- include 'dtw_warpingpaths.jinja.c' %}

{% set suffix = '_affinity_ndim' %}
{% set inner_dist = 'euclidean' %}
{%- include 'dtw_warpingpaths.jinja.c' %}


{% set suffix = '_affinity' %}
{%- include 'dtw_expandwps.jinja.c' %}


void dtw_wps_negativize_value(DTWWps* p, seq_t *wps, idx_t l1, idx_t l2, idx_t r, idx_t c) {
    idx_t idx = dtw_wps_loc(p, r, c, l1, l2);
    if (idx == 0) {
        return;
    }
    if (wps[idx] > 0 && wps[idx] != INFINITY) {
        wps[idx] = -wps[idx];
    }
}

void dtw_wps_positivize_value(DTWWps* p, seq_t *wps, idx_t l1, idx_t l2, idx_t r, idx_t c) {
    idx_t idx = dtw_wps_loc(p, r, c, l1, l2);
    if (idx == 0) {
        return;
    }
    if (wps[idx] < 0 && wps[idx] != -INFINITY) {
        wps[idx] = -wps[idx];
    }
}



/*!
 Negate the values in the warping paths matrix for the rows [rb:re].
 
 This can be used to cancel out values for an affinity matrix without losing these values.
 
 @param wps Warping paths matrix
 @param rb Row begin
 @param re Row end
 */
void dtw_wps_negativize(DTWWps* p, seq_t *wps, idx_t l1, idx_t l2, idx_t rb, idx_t re, idx_t cb, idx_t ce) {
    idx_t i, j, wpsi, cbp, cep, cbs, ces;
    idx_t idx = rb*p->width;;
    for (i=rb; i<re; i++) {
        for (j=0; j<p->width; j++) {
            if (wps[idx] > 0 && wps[idx] != INFINITY) {
                wps[idx] = -wps[idx];
            }
            idx++;
        }
    }
    // above
    for (i=1; i<rb; i++) {
        wpsi = dtw_wps_loc_columns(p, i, &cbs, &ces, l1, l2);
        /* printf("r=%zu -- [%zu,%zu]", i, cbs, ces); */
        cbp = MAX(cb, cbs);
        cep = MIN(ce, ces);
        /* printf("--> [%zu,%zu] -- %zu + %zu\n", cbp, cep, wpsi, cb-cbs); */
        idx = wpsi;
        if (cb > cbs) {
            idx += cb - cbs;
        }
        for (j=cbp; j<cep; j++) {
            if (wps[idx] > 0 && wps[idx] != INFINITY) {
                wps[idx] = -wps[idx];
            }
            idx++;
        }
    }
    // below
    for (i=re; i<l1+1; i++) {
        wpsi = dtw_wps_loc_columns(p, i, &cbs, &ces, l1, l2);
        /* printf("r=%zu -- [%zu,%zu]", i, cbs, ces); */
        cbp = MAX(cb, cbs);
        cep = MIN(ce, ces);
        if (cep - cbp == 0) {
            /* printf("break\n"); */
            break;
        }
        /* printf("--> [%zu,%zu] -- %zu + %zu\n", cbp, cep, wpsi, cb-cbs); */
        idx = wpsi;
        if (cb > cbs) {
            idx += cb - cbs;
        }
        for (j=cbp; j<cep; j++) {
            if (wps[idx] > 0 && wps[idx] != INFINITY) {
                wps[idx] = -wps[idx];
            }
            idx++;
        }
    }
}


void dtw_wps_positivize(DTWWps* p, seq_t *wps, idx_t l1, idx_t l2, idx_t rb, idx_t re, idx_t cb, idx_t ce) {
    idx_t i, j, wpsi, cbp, cep, cbs, ces;
    idx_t idx = rb*p->width;;
    for (i=rb; i<re; i++) {
        for (j=0; j<p->width; j++) {
            if (wps[idx] < 0 && wps[idx] != -INFINITY) {
                wps[idx] = -wps[idx];
            }
            idx++;
        }
    }
    // above
    for (i=1; i<rb; i++) {
        wpsi = dtw_wps_loc_columns(p, i, &cbs, &ces, l1, l2);
        /* printf("r=%zu -- [%zu,%zu]", i, cbs, ces); */
        cbp = MAX(cb, cbs);
        cep = MIN(ce, ces);
        /* printf("--> [%zu,%zu] -- %zu + %zu\n", cbp, cep, wpsi, cb-cbs); */
        idx = wpsi + (cb - cbs);
        for (j=cbp; j<cep; j++) {
            if (wps[idx] < 0 && wps[idx] != INFINITY) {
                wps[idx] = -wps[idx];
            }
            idx++;
        }
    }
    // below
    for (i=re; i<l1+1; i++) {
        wpsi = dtw_wps_loc_columns(p, i, &cbs, &ces, l1, l2);
        /* printf("r=%zu -- [%zu,%zu]", i, cbs, ces); */
        cbp = MAX(cb, cbs);
        cep = MIN(ce, ces);
        if (cep - cbp == 0) {
            /* printf("break\n"); */
            break;
        }
        /* printf("--> [%zu,%zu] -- %zu + %zu\n", cbp, cep, wpsi, cb-cbs); */
        idx = wpsi + (cb - cbs);
        for (j=cbp; j<cep; j++) {
            if (wps[idx] < 0 && wps[idx] != INFINITY) {
                wps[idx] = -wps[idx];
            }
            idx++;
        }
    }
}


/*!
Compute the location in the compact matrix from the row and column in
the implied matrix.

 @param p Settings
 @param r Row index
 @param c Column index
 @param l1 Length of first series.
 @param l2 Length of second series.
 @return Index in array representing the compact matrix.
    Returning a 0 means that location does not exist.
 */
idx_t dtw_wps_loc(DTWWps* p, idx_t r, idx_t c, idx_t l1, idx_t l2) {
    idx_t ri, ci, wpsi, wpsi_start;
    idx_t ri_width = p->width;
    idx_t min_ci, max_ci;

    // First row is inf
    ri_width = p->width;

    // A.
    min_ci = 0;
    max_ci = p->window + p->ldiffc + 1;
    for (ri=1; ri<p->ri1+1; ri++) {
        ci = min_ci;
        wpsi = 0;
        if (ri == r && ci > c) {
            printf("WARNING: dtw_wps_loc: location does not exist: %zu, %zu\n", r, c);
            return 0;
        }
        for (; ci<max_ci; ci++) {
            if (ri == r && ci == c) {
                return ri_width + wpsi;
            }
            wpsi++;
        }
        if (ri == r && ci < c) {
            printf("WARNING: dtw_wps_loc: location does not exist: %zu, %zu\n", r, c);
            return 0;
        }
        max_ci++;
        ri_width += p->width;
    }

    // B.
    min_ci = 0;
    max_ci = l2 + 1;
    for (ri=p->ri1+1; ri<p->ri2+1; ri++) {
        wpsi = 0;
        ci = min_ci;
        if (ri == r && ci > c) {
            printf("WARNING: dtw_wps_loc: location does not exist: %zu, %zu\n", r, c);
            return 0;
        }
        for (; ci<max_ci; ci++) {
            if (ri == r && ci == c) {
                return ri_width + wpsi;
            }
            wpsi++;
        }
        if (ri == r && ci < c) {
            printf("WARNING: dtw_wps_loc: location does not exist: %zu, %zu\n", r, c);
            return 0;
        }
        ri_width += p->width;
    }

    // C.
    min_ci = 1;
    max_ci = 1 + 2 * p->window - 1 + p->ldiff + 1;
    for (ri=p->ri2+1; ri<p->ri3+1; ri++) {
        ci = min_ci;
        wpsi = 0;
        if (ri == r && ci > c) {
            printf("WARNING: dtw_wps_loc: location does not exist: %zu, %zu\n", r, c);
            return 0;
        }
        for (; ci<max_ci; ci++) {
            if (ri == r && ci == c) {
                return ri_width + wpsi;
            }
            wpsi++;
        }
        if (ri == r && ci < c) {
            printf("WARNING: dtw_wps_loc: location does not exist: %zu, %zu\n", r, c);
            return 0;
        }
        min_ci++;
        max_ci++;
        ri_width += p->width;
    }

    // D.
    min_ci = MAX(0, p->ri3 + 1 - p->window - p->ldiff);
    max_ci = l2 + 1;
    wpsi_start = 2;
    if (p->ri2 == p->ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
    } else {
        min_ci = 1 + p->ri3 - p->ri2;
    }
    for (ri=p->ri3+1; ri<l1+1; ri++) {
        ci = min_ci;
        wpsi = wpsi_start - 1;
        if (ri == r && ci > c) {
            printf("WARNING: dtw_wps_loc: location does not exist: %zu, %zu\n", r, c);
            return 0;
        }
        for (; ci<max_ci; ci++) {
            if (ri == r && ci == c) {
                return ri_width + wpsi;
            }
            wpsi++;
        }
        if (ri == r && ci < c) {
            printf("WARNING: dtw_wps_loc: location does not exist: %zu, %zu\n", r, c);
            return 0;
        }
        wpsi_start++;
        min_ci++;
        ri_width += p->width;
    }

    return 0;
}


idx_t dtw_wps_loc_columns(DTWWps* p, idx_t r, idx_t *cb, idx_t *ce, idx_t l1, idx_t l2) {
    idx_t ri, wpsi, wpsi_start;
    idx_t ri_width = p->width;
    idx_t min_ci, max_ci;

    // First row is inf
    ri_width = p->width;

    // A.
    min_ci = 0;
    max_ci = p->window + p->ldiffc + 1;
    for (ri=1; ri<p->ri1+1; ri++) {
        if (ri == r) {
            *cb = min_ci;
            *ce = max_ci;
            return ri_width;
        }
        max_ci++;
        ri_width += p->width;
    }

    // B.
    min_ci = 0;
    max_ci = l2 + 1;
    for (ri=p->ri1+1; ri<p->ri2+1; ri++) {
        if (ri == r) {
            *cb = min_ci;
            *ce = max_ci;
            return ri_width;
        }
        ri_width += p->width;
    }

    // C.
    min_ci = 1;
    max_ci = 1 + 2 * p->window - 1 + p->ldiff + 1;
    for (ri=p->ri2+1; ri<p->ri3+1; ri++) {
        if (ri == r) {
            *cb = min_ci;
            *ce = max_ci;
            return ri_width;
        }
        min_ci++;
        max_ci++;
        ri_width += p->width;
    }

    // D.
    min_ci = MAX(0, p->ri3 + 1 - p->window - p->ldiff);
    max_ci = l2 + 1;
    wpsi_start = 2;
    if (p->ri2 == p->ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
    } else {
        min_ci = 1 + p->ri3 - p->ri2;
    }
    for (ri=p->ri3+1; ri<l1+1; ri++) {
        wpsi = wpsi_start - 1;
        if (ri == r) {
            *cb = min_ci;
            *ce = max_ci;
            return ri_width + wpsi;
        }
        wpsi_start++;
        min_ci++;
        ri_width += p->width;
    }

    return 0;
}


/*!
Get maximal value in matrix

 @param p Settings
 @param r Row index returned
 @param c Column index returned
 @param l1 Length of first series.
 @param l2 Length of second series.
 @return Index in array representing the compact matrix.
    Returning a 0 means that location does not exist.
 */
idx_t dtw_wps_max(DTWWps* p, seq_t *wps, idx_t *r, idx_t *c, idx_t l1, idx_t l2) {
    idx_t ri, ci, wpsi, wpsi_start;
    idx_t ri_width = p->width;
    idx_t min_ci, max_ci;
    seq_t maxval = 0;
    idx_t maxidx = 0;
    idx_t maxr = 0;
    idx_t maxc = 0;

    // First row is inf
    ri_width = p->width;

    // A.
    min_ci = 0;
    max_ci = p->window + p->ldiffc + 1;
    for (ri=1; ri<p->ri1+1; ri++) {
        ci = min_ci;
        wpsi = 0;
        for (; ci<max_ci; ci++) {
            if (wps[ri_width + wpsi] > maxval) {
                maxval = wps[ri_width + wpsi];
                maxidx = ri_width + wpsi;
                maxr = ri;
                maxc = ci;
            }
            wpsi++;
        }
        max_ci++;
        ri_width += p->width;
    }

    // B.
    min_ci = 0;
    max_ci = l2 + 1;
    for (ri=p->ri1+1; ri<p->ri2+1; ri++) {
        wpsi = 0;
        ci = min_ci;
        for (; ci<max_ci; ci++) {
            if (wps[ri_width + wpsi] > maxval) {
                maxval = wps[ri_width + wpsi];
                maxidx = ri_width + wpsi;
                maxr = ri;
                maxc = ci;
            }
            wpsi++;
        }
        ri_width += p->width;
    }

    // C.
    min_ci = 1;
    max_ci = 1 + 2 * p->window - 1 + p->ldiff + 1;
    for (ri=p->ri2+1; ri<p->ri3+1; ri++) {
        ci = min_ci;
        wpsi = 0;
        for (; ci<max_ci; ci++) {
            if (wps[ri_width + wpsi] > maxval) {
                maxval = wps[ri_width + wpsi];
                maxidx = ri_width + wpsi;
                maxr = ri;
                maxc = ci;
            }
            wpsi++;
        }
        min_ci++;
        max_ci++;
        ri_width += p->width;
    }

    // D.
    min_ci = MAX(0, p->ri3 + 1 - p->window - p->ldiff);
    max_ci = l2 + 1;
    wpsi_start = 2;
    if (p->ri2 == p->ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
    } else {
        min_ci = 1 + p->ri3 - p->ri2;
    }
    for (ri=p->ri3+1; ri<l1+1; ri++) {
        ci = min_ci;
        wpsi = wpsi_start - 1;
        for (; ci<max_ci; ci++) {
            if (wps[ri_width + wpsi] > maxval) {
                maxval = wps[ri_width + wpsi];
                maxidx = ri_width + wpsi;
                maxr = ri;
                maxc = ci;
            }
            wpsi++;
        }
        wpsi_start++;
        min_ci++;
        ri_width += p->width;
    }

    *r = maxr;
    *c = maxc;
    return maxidx;
}


{% set suffix = '' %}
{% set use_isclose = 0 %}
{%- include 'dtw_bestpath.jinja.c' %}

{% set suffix = '' %}
{% set use_isclose = 1 %}
{%- include 'dtw_bestpath.jinja.c' %}

{% set suffix = '_affinity' %}
{% set use_isclose = 0 %}
{%- include 'dtw_bestpath.jinja.c' %}


void dtw_srand(unsigned int seed) {
    if (seed == 0) {
        seed = (unsigned int)time(NULL);
    }
    // default for srand is 1
    srand(seed);
}


/*!
Sample a likely best path between two series.
 
 @param wps Array of length `(l1+1)*min(l2+1, abs(l1-l2) + 2*window-1)` with the warping paths.
 @param i1 Array of length l1+l2 to store the indices for the first sequence.
    Reverse ordered, last one is if i1 or i2 is zero.
 @param i2 Array of length l1+l2 to store the indices for the second sequence.
    Reverse ordered, last one is if i1 or i2 is zero.
 @param l1 Length of first array.
 @param l2 Length of second array.
 @param avg Average value for difference in values (order of magnitude to decide probabilities)
 @param settings for Dynamic Time Warping.
 @return length of path
 */
idx_t dtw_best_path_prob(seq_t *wps, idx_t *i1, idx_t *i2, idx_t l1, idx_t l2, seq_t avg, DTWSettings *settings) {
    DTWWps p = dtw_wps_parts(l1, l2, settings);
    
    idx_t i = 0;
    idx_t rip = l1;
    idx_t cip = l2;
    idx_t ri_widthp = p.width * (l1 - 1);
    idx_t ri_width = p.width * l1;
    idx_t min_ci;
    idx_t wpsi_start, wpsi;
    float probs[3];
    float probs_sum;
    float rnum;
    seq_t prev;
    seq_t min_diff;
    if (avg == 0.0) {
        avg = 1.0;
    }
    // printf("avg = %f\n", avg);
    
    // D. ri3 <= ri < l1
    min_ci = p.ri3 + 1 - p.window - p.ldiff;
    wpsi_start = 2;
    if (p.ri2 == p.ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
    } else {
        min_ci = 1 + p.ri3 - p.ri2;
    }
    wpsi = wpsi_start + (l2 - min_ci) - 1;
    while (rip > p.ri3 && cip > 0) {
        if (wps[ri_width + wpsi] != -1) {
            i1[i] = rip - 1;
            i2[i] = cip - 1;
            i++;
        }
        prev = wps[ri_width + wpsi];
        probs[0] = prev - wps[ri_widthp + wpsi - 1]; // Diagonal
        probs[1] = prev - wps[ri_width + wpsi - 1];  // Left
        probs[2] = prev - wps[ri_widthp + wpsi];     // Right
        min_diff = MAX3(probs[0], probs[1], probs[2]);
        if (min_diff < 0) {  min_diff = 0; }
        probs[0] = 1.0 / (avg + min_diff - probs[0]);
        probs[1] = 1.0 / (avg + min_diff - probs[1]);
        probs[2] = 1.0 / (avg + min_diff - probs[2]);
        probs_sum = probs[0] + probs[1] + probs[2];
        probs[2] = 1.0;
        probs[1] = (probs[0] + probs[1]) / probs_sum;
        probs[0] = probs[0] / probs_sum;
        rnum = (float)(rand()%1000) / 1000.0; // Never select 1.0 to not select cumulative prob of 1.0
        // printf("Probs = [%f, %f, %f] , rnum=%f\n", probs[0], probs[1], probs[2], rnum);
        // printf("%f, %f, %f, %f\n", prev, wps[ri_widthp + wpsi - 1], wps[ri_width + wpsi - 1], wps[ri_widthp + wpsi]);

        if (rnum < probs[0]) {
            // Go diagonal
            cip--;
            rip--;
            wpsi--;
            ri_width = ri_widthp;
            ri_widthp -= p.width;
        } else if (rnum < probs[1]) {
            // Go left
            cip--;
            wpsi--;
        } else {
            // Go up
            rip--;
            ri_width = ri_widthp;
            ri_widthp -= p.width;
        }
    }
    
    // C. ri2 <= ri < ri3
    while (rip > p.ri2 && cip > 0) {
        if (wps[ri_width + wpsi] != -1) {
            i1[i] = rip - 1;
            i2[i] = cip - 1;
            i++;
        }
        prev = wps[ri_width + wpsi];
        probs[0] = prev - wps[ri_widthp + wpsi];     // Diagonal
        probs[1] = prev - wps[ri_width + wpsi - 1];  // Left
        probs[2] = prev - wps[ri_widthp + wpsi + 1]; // Right
        min_diff = MAX3(probs[0], probs[1], probs[2]);
        if (min_diff < 0) {  min_diff = 0; }
        probs[0] = 1.0 / (avg + min_diff - probs[0]);
        probs[1] = 1.0 / (avg + min_diff - probs[1]);
        probs[2] = 1.0 / (avg + min_diff - probs[2]);
        probs_sum = probs[0] + probs[1] + probs[2];
        probs[2] = 1.0;
        probs[1] = (probs[0] + probs[1]) / probs_sum;
        probs[0] = probs[0] / probs_sum;
        rnum = (float)(rand()%1000) / 1000.0;
        // printf("Probs = [%f, %f, %f] , rnum=%f\n", probs[0], probs[1], probs[2], rnum);
        // printf("%f, %f, %f, %f\n", prev, wps[ri_widthp + wpsi], wps[ri_width + wpsi - 1], wps[ri_widthp + wpsi + 1]);
        
        if (rnum < probs[0]) {
            // Go diagonal
            cip--;
            rip--;
            ri_width = ri_widthp;
            ri_widthp -= p.width;
        } else if (rnum < probs[1]) {
            // Go left
            cip--;
            wpsi--;
        } else {
            // Go up
            rip--;
            wpsi++;
            ri_width = ri_widthp;
            ri_widthp -= p.width;
        }
    }
    
    // A-B. 0 <= ri < ri2
    while (rip > 0 && cip > 0) {
        if (wps[ri_width + wpsi] != -1) {
            i1[i] = rip - 1;
            i2[i] = cip - 1;
            i++;
        }
        prev = wps[ri_width + wpsi];
        probs[0] = prev - wps[ri_widthp + wpsi - 1]; // Diagonal
        probs[1] = prev - wps[ri_width  + wpsi - 1]; // Left
        probs[2] = prev - wps[ri_widthp + wpsi];     // Right
        min_diff = MAX3(probs[0], probs[1], probs[2]);
        if (min_diff < 0) {  min_diff = 0; }
        probs[0] = 1.0 / (avg + min_diff - probs[0]);
        probs[1] = 1.0 / (avg + min_diff - probs[1]);
        probs[2] = 1.0 / (avg + min_diff - probs[2]);
        probs_sum = probs[0] + probs[1] + probs[2];
        probs[2] = 1.0;
        probs[1] = (probs[0] + probs[1]) / probs_sum;
        probs[0] = probs[0] / probs_sum;
        rnum = (float)(rand()%1000) / 1000.0;
        // printf("Probs = [%f, %f, %f] , rnum=%f\n", probs[0], probs[1], probs[2], rnum);
        // printf("prev=%f, dists=%f, %f, %f, min_diff=%f, avg=%f\n",
        //        prev, wps[ri_widthp + wpsi - 1], wps[ri_width  + wpsi - 1], wps[ri_widthp + wpsi], min_diff, avg);
        
        if (rnum < probs[0]) {
            // Go diagonal
            cip--;
            rip--;
            wpsi--;
            ri_width = ri_widthp;
            ri_widthp -= p.width;
        } else {
            if (rnum < probs[1]) {
                // Go left
                cip--;
                wpsi--;
            } else {
                // Go up
                rip--;
                ri_width = ri_widthp;
                ri_widthp -= p.width;
            }
        }
    }
    return i;
}

/*!
 Compute warping path between two sequences.
 
 @param from_s First sequence
 @param from_l Length of first sequence
 @param to_s Second sequence
 @param to_l Length of second sequence
 @param from_i Stores warping path indices for the first sequence
 @param to_i Stores warping path indices for the second sequence
 @param length_i Stores resulting path  length for from_i and to_i
 @param settings Settings object
 @return distance
 */
seq_t dtw_warping_path(seq_t *from_s, idx_t from_l, seq_t* to_s, idx_t to_l, idx_t *from_i, idx_t *to_i, idx_t * length_i, DTWSettings * settings) {
    return dtw_warping_path_ndim(from_s, from_l, to_s, to_l, from_i, to_i, length_i, 1, settings);
}

seq_t dtw_warping_path_ndim(seq_t *from_s, idx_t from_l, seq_t* to_s, idx_t to_l, idx_t *from_i, idx_t *to_i, idx_t * length_i, int ndim, DTWSettings * settings) {
    idx_t wps_length = dtw_settings_wps_length(from_l, to_l, settings);
    seq_t *wps = (seq_t *)malloc(wps_length * sizeof(seq_t));
    seq_t d;
    if (settings->inner_dist == 1) {
        d = dtw_warping_paths_ndim_euclidean(wps, from_s, from_l, to_s, to_l, true, false, true,                        ndim, settings);
    } else {
        d = dtw_warping_paths_ndim(wps, from_s, from_l, to_s, to_l, true, false, true,                        ndim, settings);
        d = sqrt(d);
    }
    *length_i = dtw_best_path(wps, from_i, to_i, from_l, to_l, settings);
    free(wps);
    return d;
}

/*!
 Sample probabilistically warping path between two sequences.
 
 @return length of path
 */
seq_t dtw_warping_path_prob_ndim(seq_t *from_s, idx_t from_l, seq_t* to_s, idx_t to_l, idx_t *from_i, idx_t *to_i, idx_t *length_i, seq_t avg, int ndim, DTWSettings * settings) {
    idx_t wps_length = dtw_settings_wps_length(from_l, to_l, settings);
    seq_t *wps = (seq_t *)malloc(wps_length * sizeof(seq_t));
    seq_t d = dtw_warping_paths_ndim(wps, from_s, from_l, to_s, to_l, false, false, true, ndim, settings);
    *length_i = dtw_best_path_prob(wps, from_i, to_i, from_l, to_l, avg, settings);
    free(wps);
    return d;
}


DTWWps dtw_wps_parts(idx_t l1, idx_t l2, DTWSettings * settings) {
    DTWWps parts;
    
    parts.window = settings->window;
    parts.max_step = settings->max_step;
    parts.penalty = pow(settings->penalty, 2);
    if (parts.max_step == 0) {
        parts.max_step = INFINITY;
    } else {
        parts.max_step = pow(parts.max_step, 2);
    }
    parts.max_dist = settings->max_dist; // upper bound
    if (parts.max_dist == 0) {
        parts.max_dist = INFINITY;
    } else {
        parts.max_dist = pow(parts.max_dist, 2);
    }
    
    if (l1 > l2) {
        // x x  ldiff  = 2
        // x x
        // x x  ldiffr = 2
        // x x
        //      ldiffc = 0
        parts.ldiff = l1 - l2;  // ldiff = abs(l1 - l2)
        parts.ldiffr = parts.ldiff;
        parts.ldiffc = 0;
    } else {
        // x x x x  ldiff  = 2
        // x x x x  ldiffr = 0
        //     ldiffc = 2
        parts.ldiff  = l2 - l1; // ldiff = abs(l1 - l2)
        parts.ldiffr = 0;
        parts.ldiffc = parts.ldiff;
    }
    if (parts.window == 0) {
        parts.window = MAX(l1, l2);
        parts.width = l2 + 1;
    } else {
        parts.window = MIN(parts.window, MAX(l1, l2));
        parts.width = MIN(l2 + 1, parts.ldiff + 2*parts.window + 1);
    }
    
    parts.overlap_left_ri = MIN(parts.window + parts.ldiffr, l1 + 1);
    parts.overlap_right_ri = 0;
    if ((parts.window + parts.ldiffr) <= l1) {  // Avoids being negative
        parts.overlap_right_ri = MAX(l1 + 1 - parts.window - parts.ldiffr, 0);
    }
    parts.length = (l1 + 1) * parts.width;
    
    // A.   0 <= ri < ri1
    // [0 0 x x x]
    // [0 0 0 x x]
    parts.ri1 = MIN(l1, MIN(parts.overlap_left_ri, parts.overlap_right_ri));
    // B. ri1 <= ri < ri2
    // [0 0 0 0 0]
    // [0 0 0 0 0]
    parts.ri2 = MIN(l1, parts.overlap_left_ri);
    // C. ri2 <= ri < r3
    // [x 0 0 x x]
    // [x x 0 0 x]
    parts.ri3 = MIN(l1, MAX(parts.overlap_left_ri, parts.overlap_right_ri));
    // D. ri3 <= ri < l1
    // [x 0 0 0 0]
    // [x x 0 0 0]
    
//    printf("l1=%zu, l2=%zu, window=%zu, width=%zu, length=%zu\n", l1, l2, parts.window, parts.width, parts.length);
//    printf("ldiffc=%zu, ldiffr=%zu\n", parts.ldiffc, parts.ldiffr);
//    printf("overlap_left_ri=%zu, overlap_right_ri=%zu\n", parts.overlap_left_ri, parts.overlap_right_ri);
//    printf("ri1=%zu, ri2=%zu, ri3=%zu\n", parts.ri1, parts.ri2, parts.ri3);
    
    return parts;
}


// MARK: Bounds

/*!
 Euclidean upper bound for DTW.
 
 @see ed.euclidean_distance.
 */
seq_t ub_euclidean(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2) {
    return euclidean_distance(s1, l1, s2, l2);
}


/*!
 Euclidean upper bound for DTW.
 
 @see ed.euclidean_distance_ndim.
*/
seq_t ub_euclidean_ndim(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim) {
    return euclidean_distance_ndim(s1, l1, s2, l2, ndim);
}

/*!
 Euclidean upper bound for DTW.
 
 @see ed.euclidean_distance.
 */
seq_t ub_euclidean_euclidean(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2) {
    return euclidean_distance_euclidean(s1, l1, s2, l2);
}


/*!
 Euclidean upper bound for DTW.
 
 @see ed.euclidean_distance_ndim.
*/
seq_t ub_euclidean_ndim_euclidean(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim) {
    return euclidean_distance_ndim_euclidean(s1, l1, s2, l2, ndim);
}


{% set inner_dist = 'squaredeuclidean' %}
{%- include 'lb_keogh.jinja.c' %}

{% set inner_dist = 'euclidean' %}
{%- include 'lb_keogh.jinja.c' %}


// MARK: Block

/* Create settings struct with default values (all extras deactivated). */
DTWBlock dtw_block_empty(void) {
    DTWBlock b = {
        .rb = 0,  // row-begin
        .re = 0,  // row-end
        .cb = 0,  // column-begin
        .ce = 0,  // column-end
        .triu = true // only fill upper triangular marix
    };
    return b;
}


void dtw_block_print(DTWBlock *block) {
    printf("DTWBlock {\n");
    printf("  rb = %zu\n", block->rb);
    printf("  re = %zu\n", block->re);
    printf("  cb = %zu\n", block->cb);
    printf("  ce = %zu\n", block->ce);
    printf("  triu = %s\n", block->triu ? "true" : "false");
    printf("}\n");
}


bool dtw_block_is_valid(DTWBlock *block, idx_t nb_series_r, idx_t nb_series_c) {
    if (block->rb >= block->re) {
        printf("ERROR: Block row range is 0 or smaller\n");
        return false;
    }
    if (block->cb >= block->ce) {
        printf("ERROR: Block row range is 0 or smaller\n");
        return false;
    }
    if (block->rb >= nb_series_r) {
        printf("ERROR: Block rb exceeds number of series\n");
        return false;
    }
    if (block->re > nb_series_r) {
        printf("ERROR: Block re exceeds number of series\n");
        return false;
    }
    if (block->cb >= nb_series_c) {
        printf("ERROR: Block cb exceeds number of series\n");
        return false;
    }
    if (block->ce > nb_series_c) {
        printf("ERROR: Block ce exceeds number of series\n");
        return false;
    }
    return true;
}


// MARK: Distance Matrix


{% set suffix = 'ptrs' %}
{%- include 'dtw_distances.jinja.c' %}

{% set suffix = 'matrix' %}
{%- include 'dtw_distances.jinja.c' %}

{% set suffix = 'ndim_matrix' %}
{%- include 'dtw_distances.jinja.c' %}

{% set suffix = 'ndim_ptrs' %}
{%- include 'dtw_distances.jinja.c' %}

{% set suffix = 'matrices' %}
{%- include 'dtw_distances.jinja.c' %}

{% set suffix = 'ndim_matrices' %}
{%- include 'dtw_distances.jinja.c' %}

idx_t dtw_distances_length(DTWBlock *block, idx_t nb_series_r, idx_t nb_series_c) {
    // Note: int is usually 32-bit even on 64-bit systems
    idx_t ir;
    idx_t length = 0;  // Should be sidx_t but not available on all platforms
    idx_t overflow_buffer, delta;
    idx_t max_nb_series;

    if (block == NULL || block->re == 0 || block->ce == 0) {
        // Check for overflow
        max_nb_series = idx_t_max / nb_series_r;
        if (nb_series_c > max_nb_series) {
            printf("ERROR: Length of array needed to represent the distance matrix for (%zu x %zu) series is larger than the maximal value allowed (unsigned %zu)\n", nb_series_c, nb_series_r, idx_t_max);
            return 0;
        }
        if (block != NULL && block->triu) {
            if (nb_series_r == nb_series_c) {
                // First divide the even number to avoid overflowing
                if (nb_series_r % 2 == 0) {
                    length = (nb_series_r / 2) * (nb_series_r - 1);
                } else {
                    length = nb_series_r * ((nb_series_r - 1) / 2);
                }
            } else {
                if (nb_series_r > nb_series_c) {
                    // First divide the even number to avoid overflowing
                    if (nb_series_c % 2 == 0) {
                        length = (nb_series_c / 2) * (nb_series_c - 1);
                    } else {
                        length = nb_series_c * ((nb_series_c - 1) / 2);
                    }
                }
                if (nb_series_r < nb_series_c) {
                    // First divide the even number to avoid overflowing
                    if (nb_series_c % 2 == 0) {
                        length = (nb_series_c / 2) * (nb_series_c - 1);
                    } else {
                        length = nb_series_c * ((nb_series_c - 1) / 2);
                    }
                    nb_series_c -= nb_series_r;
                    if (nb_series_c % 2 == 0) {
                        length -= (nb_series_c / 2) * (nb_series_c - 1);
                    } else {
                        length -= nb_series_c * ((nb_series_c - 1) / 2);
                    }
                }
            }
        } else { // triu=false
            length = nb_series_c * nb_series_r;
        }
    } else {
        if (!dtw_block_is_valid(block, nb_series_r, nb_series_c)) {
            return 0;
        }
        if (block->triu) {
            for (ir=block->rb; ir<block->re; ir++) {
                if (ir < block->cb) {
                    delta = block->ce - block->cb;
                } else { // ir >= block->cb
                    if (block->ce <= ir) {
                        // ir only increases so block->ce will always be < ir
                        // delta = 0
                        break;
                    } else { // block->ce > ir
                        delta = block->ce - ir - 1;
                    }
                }
                overflow_buffer = idx_t_max - length;
                if (overflow_buffer < delta) {
                    printf("Trying to execute %zu + %zu > %zu\n", length, delta, idx_t_max);
                    {%- raw %}
                    printf("ERROR: Length of array needed to represent the distance matrix for %zu x %zu series and block {%zu, %zu, %zu, %zu} is larger than the maximal value allowed (unsigned %zu)\n", nb_series_r, nb_series_c, block->rb, block->re, block->cb, block->ce, idx_t_max);
                    {%- endraw %}
                    return 0;
                }
                length += delta;
            }
        } else { // triu=false
            // Check for overflow
            max_nb_series = idx_t_max / (block->re - block->rb);
            if ((block->ce - block->cb) > max_nb_series) {
                printf("ERROR: Length of array needed to represent the distance matrix for block ");
                printf("(%zd x %zd) is larger than the maximal value allowed (unsigned %zd)\n",
                        (block->re - block->rb), (block->ce - block->cb), idx_t_max);
                return 0;
            }
            length = (block->re - block->rb) * (block->ce - block->cb);
        }
    }
    return length;
}

// MARK: DBA

{% set suffix = 'ptrs' %}
{%- include 'dtw_dba.jinja.c' %}

{% set suffix = 'matrix' %}
{%- include 'dtw_dba.jinja.c' %}

// MARK: Auxiliary functions

/*! Interupt Handler */
void dtw_int_handler(int dummy) {
    printf("Interrupt process, stopping ...\n");
    keepRunning = 0;
}

void dtw_printprecision_set(int precision) {
    printPrecision = precision;
}

void dtw_printprecision_reset(void) {
    printPrecision = 3;
}

/* Helper function for debugging. */
void dtw_print_wps_compact(seq_t * wps, idx_t l1, idx_t l2, DTWSettings* settings) {
    DTWWps p = dtw_wps_parts(l1, l2, settings);
    for (idx_t wpsi=0; wpsi<p.width; wpsi++) {
        dtw_print_nb(wps[wpsi]);
    }
    printf("\n");
    for (idx_t ri=0; ri<l1; ri++) {
        for (idx_t wpsi=0; wpsi<p.width; wpsi++) {
            dtw_print_nb(wps[(ri+1)*p.width+wpsi]);
        }
        if (ri < p.ri1) { printf("  # a %zu", ri); }
        if (p.ri1 <= ri && ri < p.ri2) { printf("  # b %zu", ri); }
        if (p.ri2 <= ri && ri < p.ri3) { printf("  # c %zu", ri); }
        if (p.ri3 <= ri) { printf("  # d %zu", ri); }
        printf("\n");
    }
}

void dtw_print_wps(seq_t * wps, idx_t l1, idx_t l2, DTWSettings* settings) {
    DTWWps p = dtw_wps_parts(l1, l2, settings);
    
    idx_t ri, ci, min_ci, max_ci, wpsi, wpsi_start;
    
    // Top row: ri = -1
    printf(" [[ ");
    dtw_print_nb(wps[0]);
    printf(" ");
    wpsi = 1;
    for (ci=0; ci<MIN(p.window + p.ldiffc, l2); ci++) {
        dtw_print_nb(wps[wpsi]);
        printf(" ");
        wpsi++;
    }
    for (; wpsi<p.width; wpsi++) {
        dtw_print_nb(wps[wpsi]);
        printf("_");
        ci++;
    }
    for (; ci<l2; ci++) {
        printf(" ");
        dtw_print_ch("inf.");
    }
    printf("]\n");
    
    // A. Rows: 0 <= ri < min(overlap_left_ri, overlap_right_ri)
    min_ci = 0;
    max_ci = p.window + p.ldiffc; // ri < overlap_right_i
    for (ri=0; ri<p.ri1; ri++) {
        printf("  [ ");
        dtw_print_nb(wps[p.width*(ri + 1)]); // wpsi = 0
        printf("_");
        wpsi = 1;
        for (ci=min_ci; ci<max_ci; ci++) {
            dtw_print_nb(wps[(ri+1)*p.width + wpsi]);
            printf(" ");
            //printf("%zux%zu   ", wpsi, ci);
            wpsi++;
        }
        for (; wpsi<p.width; wpsi++) {
            dtw_print_nb(wps[(ri+1)*p.width + wpsi]);
            printf("_");
            ci++;
        }
        for (; ci<l2 ;ci++) {
            dtw_print_ch(".inf");
            printf(" ");
        }
        printf("],  # a %zu\n", ri);
        max_ci++;
    }
    
    // B. Rows: min(overlap_left_ri, overlap_right_ri) <= ri < overlap_left_ri
    min_ci = 0;
    max_ci = l2; // ri >= overlap_right_i
    for (ri=p.ri1; ri<p.ri2; ri++) {
        printf("  [ ");
        dtw_print_nb(wps[p.width*(ri + 1)]);
        printf("_");
        wpsi = 1;
        for (ci=min_ci; ci<max_ci; ci++) {
            dtw_print_nb(wps[(ri+1)*p.width + wpsi]);
            printf(" ");
            wpsi++;
        }
        for (; wpsi<p.width; wpsi++) {
            dtw_print_nb(wps[(ri+1)*p.width + wpsi]);
            printf("_");
            ci++;
        }
        for (; ci<l2 ;ci++) {
            dtw_print_ch(".inf");
            printf(" ");
        }
        printf("],  # b %zu\n", ri);
    }
    
    // C. Rows: overlap_left_ri <= ri < MAX(parts.overlap_left_ri, parts.overlap_right_ri)
    min_ci = 1;
    max_ci = 1 + 2 * p.window - 1 + p.ldiff;
    for (ri=p.ri2; ri<p.ri3; ri++) {
        printf("  [ ");
        for (ci=0; ci<min_ci ;ci++) {
            dtw_print_ch(".inf");
            printf(" ");
        }
        dtw_print_nb(wps[(ri+1)*p.width + 0]);
        printf("_");
        wpsi = 1;
        for (ci=min_ci; ci<max_ci; ci++) {
            dtw_print_nb(wps[(ri+1)*p.width + wpsi]);
            printf(" ");
            wpsi++;
        }
        for (; wpsi<p.width && ci<l2; wpsi++) {
            dtw_print_nb(wps[(ri+1)*p.width + wpsi]);
            printf("_");
            ci++;
        }
        for (; ci<l2 ;ci++) {
            dtw_print_ch(".inf");
            printf(" ");
        }
        printf("],  # c %zu\n", ri);
        min_ci++;
        max_ci++;
    }
    
    // D. Rows: MAX(overlap_left_ri, overlap_right_ri) < ri <= l1
    min_ci = p.ri3 + 1 - p.window - p.ldiff;
    wpsi_start = 2;
    if (p.ri2 == p.ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
    } else {
        min_ci = 1 + p.ri3 - p.ri2;
    }
    for (ri=p.ri3; ri<l1; ri++) {
        printf("  [ ");
        if (p.ri2 == p.ri3) {
            // C is skipped
            for (wpsi = 0; wpsi<wpsi_start; wpsi++) {
                dtw_print_nb(wps[(ri+1)*p.width + wpsi]);
                printf("_");
            }
            ci = wpsi_start - 1;
        } else {
            dtw_print_ch(".inf");
            printf(" ");
            for (ci=0; ci<(min_ci - wpsi_start) ;ci++) {
                dtw_print_ch(".inf");
                printf(" ");
            }
            for (wpsi = 0; wpsi<wpsi_start; wpsi++) {
                dtw_print_nb(wps[(ri+1)*p.width + wpsi]);
                printf("_");
                ci++;
            }
        }
        assert(ci == min_ci);
        assert(wpsi == wpsi_start);
        wpsi = wpsi_start;
        for (ci=min_ci; ci<l2; ci++) {
            dtw_print_nb(wps[(ri+1)*p.width + wpsi]);
            printf(" ");
            wpsi++;
        }
        if (ri == l1 - 1) {
            printf("]]  # d %zu\n", ri);
        } else {
            printf("],  # d %zu\n", ri);
        }
        min_ci++;
        wpsi_start++;
    }
}

/* Helper function for debugging. */
void dtw_print_twoline(seq_t * dtw, idx_t r, idx_t c, idx_t length, int i0, int i1, idx_t skip, idx_t skipp, idx_t maxj, idx_t minj) {
    char buffer[20];
    char format[5];
    snprintf(format, sizeof(format), "%%.%df ", printPrecision);
    idx_t ci_cor; // corrected column index
    // Row 1
    printf("[[ ");
    for (idx_t ci=0; ci<c; ci++) {
        if (ci < maxj || ci > minj) {
            printf("x ");
        } else {
            ci_cor = i0*length + ci - skipp;
            snprintf(buffer, sizeof(buffer), format, dtw[ci_cor]);
            printf("%-*s", printPrecision + 3, buffer);
        }
    }
    printf("]\n");
    // Row 2
    printf(" [ ");
    for (idx_t ci=0; ci<c; ci++) {
        if (ci < maxj || ci > minj) {
            printf("x ");
        } else {
            ci_cor = i1*length + ci - skip;
            snprintf(buffer, sizeof(buffer), format, dtw[ci_cor]);
            printf("%-*s", printPrecision + 3, buffer);
        }
    }
    printf("]]\n");
}

void dtw_print_nb(seq_t value) {
    snprintf(printFormat, sizeof(printFormat), "%%.%df", printPrecision);
    snprintf(printBuffer, sizeof(printBuffer), printFormat, value);
    printf("%*s", printDigits, printBuffer);
    // "%-*s" would left align
}

void dtw_print_ch(char* string) {
    printf("%*s", printDigits, string);
    // "%-*s" would left align
}


