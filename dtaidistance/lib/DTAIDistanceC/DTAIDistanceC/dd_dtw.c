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
        .only_ub = false
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
    printf("}\n");
}

// MARK: DTW

/**
Compute the DTW between two series.

@param s1 First sequence
@param l1 Length of first sequence
@param s2 Second sequence
@param l2 Length of second sequence
@param settings A DTWSettings struct with options for the DTW algorithm.
*/
seq_t dtw_distance(seq_t *s1, idx_t l1,
                      seq_t *s2, idx_t l2,
                      DTWSettings *settings) {
    assert(settings->psi_1b < l1 && settings->psi_1e < l1 &&
           settings->psi_2b < l2 && settings->psi_2e < l2);
    idx_t ldiff;
    idx_t dl;
    // DTWPruned
    idx_t sc = 0;
    idx_t ec = 0;
    bool smaller_found;
    idx_t ec_next;
    // signal(SIGINT, dtw_int_handler); // not compatible with OMP
    
    idx_t window = settings->window;
    seq_t max_step = settings->max_step;
    seq_t max_dist = settings->max_dist;
    seq_t penalty = settings->penalty;
    
    #ifdef DTWDEBUG
    printf("r=%zu, c=%zu\n", l1, l2);
    #endif
    if (settings->use_pruning || settings->only_ub) {
        max_dist = pow(ub_euclidean(s1, l1, s2, l2), 2);
        if (settings->only_ub) {
            return max_dist;
        }
    } else if (max_dist == 0) {
        max_dist = INFINITY;
    } else {
        max_dist = pow(max_dist, 2);
    }
    if (l1 > l2) {
        ldiff = l1 - l2;
        dl = ldiff;
    } else {
        ldiff  = l2 - l1;
        dl = 0;
    }
    if (settings->max_length_diff != 0 && ldiff > settings->max_length_diff) {
        return INFINITY;
    }
    if (window == 0) {
        window = MAX(l1, l2);
    }
    if (max_step == 0) {
        max_step = INFINITY;
    } else {
        max_step = pow(max_step, 2);
    }
    penalty = pow(penalty, 2);
    // rows is for series 1, columns is for series 2
    idx_t length = MIN(l2+1, ldiff + 2*window + 1);
    assert(length > 0);
    seq_t * dtw = (seq_t *)malloc(sizeof(seq_t) * length * 2);
    if (!dtw) {
        printf("Error: dtw_distance - Cannot allocate memory (size=%zu)\n", length*2);
        return 0;
    }
    idx_t i;
    idx_t j;
    for (j=0; j<length*2; j++) {
        dtw[j] = INFINITY;
    }
    // Deal with psi-relaxation in first row
    for (i=0; i<settings->psi_2b + 1; i++) {
        dtw[i] = 0;
    }
    idx_t skip = 0;
    idx_t skipp = 0;
    int i0 = 1;
    int i1 = 0;
    idx_t minj;
    idx_t maxj;
    idx_t curidx = 0;
    idx_t dl_window = dl + window - 1;
    idx_t ldiff_window = window;
    if (l2 > l1) {
        ldiff_window += ldiff;
    }
    seq_t minv;
    seq_t d;
    seq_t tempv;
    seq_t psi_shortest = INFINITY;
    keepRunning = 1;
    for (i=0; i<l1; i++) {
        // if (!keepRunning){  // not compatible with OMP
        //     free(dtw);
        //     printf("Stop computing DTW...\n");
        //     return INFINITY;
        // }
//        maxj = i;
//        if (maxj > dl_window) {
//            maxj -= dl_window;
//        } else {
//            maxj = 0;
//        }
        maxj = (i - dl_window) * (i > dl_window);
        // No risk for overflow/modulo because we also need to store dtw of size
        // MIN(l2+1, ldiff + 2*window + 1) ?
        minj = i + ldiff_window;
        if (minj > l2) {
            minj = l2;
        }
        skipp = skip;
        skip = maxj;
        i0 = 1 - i0;
        i1 = 1 - i1;
        // Reset new line i1
        for (j=0; j<length; j++) {
            dtw[length * i1 + j] = INFINITY;
        }
//        if (length == l2 + 1) {
//            skip = 0;
//        }
        skip = skip * (length != l2 + 1);
        // PrunedDTW
        if (sc > maxj) {
            #ifdef DTWDEBUG
            printf("correct maxj to sc: %zu -> %zu (saved %zu computations)\n", maxj, sc, sc-maxj);
            #endif
            maxj = sc;
        }
        smaller_found = false;
        ec_next = i;
        // Deal with psi-relaxation in first column
        if (settings->psi_1b != 0 && maxj == 0 && i < settings->psi_1b) {
            dtw[i1*length + 0] = 0;
        }
        #ifdef DTWDEBUG
        printf("i=%zu, maxj=%zu, minj=%zu\n", i, maxj, minj);
        #endif
        for (j=maxj; j<minj; j++) {
            #ifdef DTWDEBUG
            printf("ri=%zu,ci=%zu, s1[i] = s1[%zu] = %f , s2[j] = s2[%zu] = %f\n", i, j, i, s1[i], j, s2[j]);
            #endif
            d = EDIST(s1[i], s2[j]);
            if (d > max_step) {
                // Let the value be INFINITY as initialized
                continue;
            }
            curidx = i0 * length + j - skipp;
            minv = dtw[curidx];
            curidx += 1;
            tempv = dtw[curidx] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            curidx = i1 * length + j - skip;
            tempv = dtw[curidx] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            #ifdef DTWDEBUG
            printf("d = %f, minv = %f\n", d, minv);
            #endif
            curidx += 1;
            dtw[curidx] = d + minv;
            #ifdef DTWDEBUG
            printf("%zu, %zu, %zu\n",i0*length + j - skipp,i0*length + j + 1 - skipp,i1*length + j - skip);
            printf("%f, %f, %f\n",dtw[i0*length + j - skipp],dtw[i0*length + j + 1 - skipp],dtw[i1*length + j - skip]);
            printf("i=%zu, j=%zu, d=%f, skip=%zu, skipp=%zu\n",i,j,d,skip,skipp);
            #endif
            // PrunedDTW
            if (dtw[curidx] > max_dist) {
                #ifdef DTWDEBUG
                printf("dtw[%zu] = %f > %f\n", curidx, dtw[curidx], max_dist);
                #endif
                if (!smaller_found) {
                    sc = j + 1;
                }
                if (j >= ec) {
                    #ifdef DTWDEBUG
                    printf("Break because of pruning with j=%zu, ec=%zu (saved %zu computations)\n", j, ec, minj-j);
                    #endif
                    break;
                }
            } else {
                smaller_found = true;
                ec_next = j + 1;
            }
        }
        ec = ec_next;
        // Deal with Psi-relaxation in last column
        if (settings->psi_1e != 0 && minj == l2 && l1 - 1 - i <= settings->psi_1e) {
            if (dtw[curidx] < psi_shortest) {
                // curidx is the last value
                psi_shortest = dtw[curidx];
            }
        }
        #ifdef DTWDEBUG
        dtw_print_twoline(dtw, l1, l2, length, i0, i1, skip, skipp, maxj, minj);
        #endif
    }

    if (window - 1 < 0) {
        l2 += window - 1;
    }
    seq_t result = sqrt(dtw[length * i1 + l2 - skip]);
    // Deal with psi-relaxation in the last row
    if (settings->psi_2e != 0) {
        for (i=l2 - skip - settings->psi_2e; i<l2 - skip + 1; i++) { // iterate over vci
            if (dtw[i1*length + i] < psi_shortest) {
                psi_shortest = dtw[i1*length + i];
            }
        }
        result = sqrt(psi_shortest);
    }
    free(dtw);
    // signal(SIGINT, SIG_DFL);  // not compatible with OMP
    if (settings->max_dist !=0 && result > settings->max_dist) {
        // DTWPruned keeps the last value larger than max_dist. Correct for this.
        result = INFINITY;
    }
    return result;
}

/**
 Compute the DTW between two n-dimensional series.

 @param s1 First sequence
 @param l1 Length of first sequence. In tuples, real length should be length*ndim.
 @param s2 Second sequence
 @param l2 Length of second sequence. In tuples, real length should be length*ndim.
 @param ndim Number of dimensions
 @param settings A DTWSettings struct with options for the DTW algorithm.
*/
seq_t dtw_distance_ndim(seq_t *s1, idx_t l1,
                           seq_t *s2, idx_t l2, int ndim,
                           DTWSettings *settings) {
    assert(settings->psi_1b < l1 && settings->psi_1e < l1 &&
           settings->psi_2b < l2 && settings->psi_2e < l2);
    idx_t ldiff;
    idx_t dl;
    // DTWPruned
    idx_t sc = 0;
    idx_t ec = 0;
    bool smaller_found;
    idx_t ec_next;
    signal(SIGINT, dtw_int_handler);
    
    idx_t window = settings->window;
    seq_t max_step = settings->max_step;
    seq_t max_dist = settings->max_dist;
    seq_t penalty = settings->penalty;
    
    #ifdef DTWDEBUG
    printf("r=%zu, c=%zu\n", l1, l2);
    #endif
    if (settings->use_pruning || settings->only_ub) {
        max_dist = pow(ub_euclidean_ndim(s1, l1, s2, l2, ndim), 2);
        if (settings->only_ub) {
            return max_dist;
        }
    } else if (max_dist == 0) {
        max_dist = INFINITY;
    } else {
        max_dist = pow(max_dist, 2);
    }
    if (l1 > l2) {
        ldiff = l1 - l2;
    } else {
        ldiff  = l2 - l1;
    }
    if (l1 > l2) {
        dl = ldiff;
    } else {
        dl = 0;
    }
    if (settings->max_length_diff != 0 && ldiff > settings->max_length_diff) {
        return INFINITY;
    }
    if (window == 0) {
        window = MAX(l1, l2);
    }
    if (max_step == 0) {
        max_step = INFINITY;
    } else {
        max_step = pow(max_step, 2);
    }
    penalty = pow(penalty, 2);
    idx_t length = MIN(l2+1, ldiff + 2*window + 1);
    assert(length > 0);
    seq_t * dtw = (seq_t *)malloc(sizeof(seq_t) * length * 2);
    if (!dtw) {
        printf("Error: dtw_distance - Cannot allocate memory (size=%zu)\n", length*2);
        return 0;
    }
    idx_t i;
    idx_t j;
    idx_t i_idx;
    idx_t j_idx;
    for (j=0; j<length*2; j++) {
        dtw[j] = INFINITY;
    }
    // Deal with psi-relaxation in first row
    for (i=0; i<settings->psi_2b + 1; i++) {
        dtw[i] = 0;
    }
    idx_t skip = 0;
    idx_t skipp = 0;
    int i0 = 1;
    int i1 = 0;
    idx_t minj;
    idx_t maxj;
    idx_t curidx;
    idx_t dl_window = dl + window - 1;
    idx_t ldiff_window = window;
    if (l2 > l1) {
        ldiff_window += ldiff;
    }
    seq_t minv;
    seq_t d; // DTYPE_t
    seq_t tempv;
    seq_t psi_shortest = INFINITY;
    keepRunning = 1;
    for (i=0; i<l1; i++) {
        if (!keepRunning){
            free(dtw);
            printf("Stop computing DTW...\n");
            return INFINITY;
        }
        i_idx = i * ndim;
        maxj = i;
        if (maxj > dl_window) {
            maxj -= dl_window;
        } else {
            maxj = 0;
        }
        skipp = skip;
        skip = maxj;
        i0 = 1 - i0;
        i1 = 1 - i1;
        for (j=0; j<length; j++) {
            dtw[length * i1 + j] = INFINITY;
        }
        if (length == l2 + 1) {
            skip = 0;
        }
        // No risk for overflow/modulo because we also need to store dtw of size
        // MIN(l2+1, ldiff + 2*window + 1) ?
        minj = i + ldiff_window;
        if (minj > l2) {
            minj = l2;
        }
        // PrunedDTW
        if (sc > maxj) {
            #ifdef DTWDEBUG
            printf("correct maxj to sc: %zu -> %zu (saved %zu computations)\n", maxj, sc, sc-maxj);
            #endif
            maxj = sc;
        }
        smaller_found = false;
        ec_next = i;
        // Deal with psi-relaxation in first column
        if (settings->psi_1b != 0 && maxj == 0 && i < settings->psi_1b) {
            dtw[i1*length + 0] = 0;
        }
        #ifdef DTWDEBUG
        printf("i=%zu, maxj=%zu, minj=%zu\n", i, maxj, minj);
        #endif
        for (j=maxj; j<minj; j++) {
            j_idx = j * ndim;
            #ifdef DTWDEBUG
            printf("ri=%zu,ci=%zu, s1[i] = s1[%zu] = %f , s2[j] = s2[%zu] = %f\n", i, j, i, s1[i], j, s2[j]);
            #endif
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += EDIST(s1[i_idx + d_i], s2[j_idx + d_i]);
            }
            if (d > max_step) {
                // Let the value be INFINITY as initialized
                continue;
            }
            curidx = i0*length + j - skipp;
            minv = dtw[curidx];
            curidx += 1;
            tempv = dtw[curidx] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            curidx = i1*length + j - skip;
            tempv = dtw[curidx] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            #ifdef DTWDEBUG
            printf("d = %f, minv = %f\n", d, minv);
            #endif
            curidx += 1;
            dtw[curidx] = d + minv;
            #ifdef DTWDEBUG
            printf("%zu, %zu, %zu\n",i0*length + j - skipp,i0*length + j + 1 - skipp,i1*length + j - skip);
            printf("%f, %f, %f\n",dtw[i0*length + j - skipp],dtw[i0*length + j + 1 - skipp],dtw[i1*length + j - skip]);
            printf("i=%zu, j=%zu, d=%f, skip=%zu, skipp=%zu\n",i,j,d,skip,skipp);
            #endif
            // PrunedDTW
            if (dtw[curidx] > max_dist) {
                #ifdef DTWDEBUG
                printf("dtw[%zu] = %f > %f\n", curidx, dtw[curidx], max_dist);
                #endif
                if (!smaller_found) {
                    sc = j + 1;
                }
                if (j >= ec) {
                    #ifdef DTWDEBUG
                    printf("Break because of pruning with j=%zu, ec=%zu (saved %zu computations)\n", j, ec, minj-j);
                    #endif
                    break;
                }
            } else {
                smaller_found = true;
                ec_next = j + 1;
            }
        }
        ec = ec_next;
        // Deal with Psi-relaxation in last column
        if (settings->psi_1e != 0 && minj == l2 && l1 - 1 - i <= settings->psi_1e) {
            if (dtw[(i1 + 1)*length - 1] < psi_shortest) {
                psi_shortest = dtw[(i1 + 1)*length - 1];
            }
        }
        #ifdef DTWDEBUG
        dtw_print_twoline(dtw, l1, l2, length, i0, i1, skip, skipp, maxj, minj);
        #endif
    }
    if (window - 1 < 0) {
        l2 += window - 1;
    }
    seq_t result = sqrt(dtw[length * i1 + l2 - skip]);
    // Deal with psi-relaxation in the last row
    if (settings->psi_2e != 0) {
        for (i=l2 - skip - settings->psi_2e; i<l2 - skip + 1; i++) { // iterate over vci
            if (dtw[i1*length + i] < psi_shortest) {
                psi_shortest = dtw[i1*length + i];
            }
        }
        result = sqrt(psi_shortest);
    }
    free(dtw);
    signal(SIGINT, SIG_DFL);
    if (settings->max_dist !=0 && result > settings->max_dist) {
        // DTWPruned keeps the last value larger than max_dist. Correct for this.
        result = INFINITY;
    }
    return result;
}

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

seq_t dtw_warping_paths_ndim(seq_t *wps,
                        seq_t *s1, idx_t l1,
                        seq_t *s2, idx_t l2,
                        bool return_dtw, bool do_sqrt, bool psi_neg,
                        int ndim,
                        DTWSettings *settings) {
    // DTWPruned
    idx_t sc = 0;
    idx_t ec = 0;
    idx_t ec_next;
    bool smaller_found;
    
    DTWWps p = dtw_wps_parts(l1, l2, settings);

    if (settings->use_pruning || settings->only_ub) {
        if (ndim == 1) {
            p.max_dist = pow(ub_euclidean(s1, l1, s2, l2), 2);
        } else {
            p.max_dist = pow(ub_euclidean_ndim(s1, l1, s2, l2, ndim), 2);
        }
        if (settings->only_ub) {
            if (do_sqrt) {
                return sqrt(p.max_dist);
            } else {
                return p.max_dist;
            }
        }
    }
    
    idx_t ri, ci, min_ci, max_ci, wpsi, wpsi_start;
    
    // Top row: ri = -1
    for (wpsi=0; wpsi<settings->psi_2b+1; wpsi++) {
        // ci = wpsi - 1
        wps[wpsi] = 0;
    }
    for (wpsi=settings->psi_2b+1; wpsi<p.width; wpsi++) {
        // MIN(window+ldiffc-1,l2) would be enough
        // ci = wpsi - 1
        wps[wpsi] = INFINITY;
    }
    // First column:
    wpsi = p.width;
    for (ri=0; ri<settings->psi_1b; ri++) {
        wps[wpsi] = 0;
        wpsi += p.width;
    }
    for (; ri<l1; ri++) {
        wps[wpsi] = INFINITY;
        wpsi += p.width;
    }
    
//    dtw_print_wps_compact(wps, l1, l2, settings);
//    dtw_print_wps(wps, l1, l2, settings);
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
        // PrunedDTW
        if (sc <= min_ci) {} else {
            for (; ci<sc; ci++) {
                wps[ri_width + wpsi] = INFINITY;
                wpsi++;
            }
        }
        smaller_found = false;
        ec_next = ri;
        // A region assumes wps has the same column indices in the previous row
        for (; ci<max_ci; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += EDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            if (d > p.max_step) { wps[ri_width + wpsi] = INFINITY; wpsi++; continue;}
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
            wpsi++;
        }
        ec = ec_next;
        for (idx_t i=ri_width + wpsi; i<ri_width + p.width; i++) {
            wps[i] = INFINITY;
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
        // PrunedDTW
        if (sc <= min_ci) {} else {
            for (; ci<sc; ci++) {
                wps[ri_width + wpsi] = INFINITY;
                wpsi++;
            }
        }
        smaller_found = false;
        ec_next = ri;
        for (; ci<max_ci; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += EDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            if (d > p.max_step) { wps[ri_width + wpsi] = INFINITY; wpsi++; continue;}
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
            wpsi++;
        }
        ec = ec_next;
        for (idx_t i=ri_width + wpsi; i<ri_width + p.width; i++) {
            wps[i] = INFINITY;
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
        wps[ri_width] = INFINITY;
        wpsi = 1;
        // PrunedDTW
        if (sc <= min_ci) {} else {
            for (; ci<sc; ci++) {
                wps[ri_width + wpsi] = INFINITY;
                wpsi++;
            }
        }
        smaller_found = false;
        ec_next = ri;
        for (; ci<max_ci; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += EDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            if (d > p.max_step) { wps[ri_width + wpsi] = INFINITY; wpsi++; continue;}
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
            wpsi++;
        }
        ec = ec_next;
        for (idx_t i=ri_width + wpsi; i<ri_width + p.width; i++) {
            wps[i] = INFINITY;
        }
        min_ci++;
        max_ci++;
        ri_widthp = ri_width;
        ri_width += p.width;
    }
    
    // D. Rows: MAX(overlap_left_ri, overlap_right_ri) < ri <= l1
    // [x 0 0 0 0]
    // [x x 0 0 0]
    min_ci = MAX(0, p.ri3 + 1 - p.window - p.ldiff);
    wpsi_start = 2;
    if (p.ri2 == p.ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
    }
    for (ri=p.ri3; ri<l1; ri++) {
        ri_idx = ri * ndim;
        ci = min_ci;
        wpsi = wpsi_start;
        for (idx_t i=ri_width; i<(ri_width + wpsi); i++) {
            wps[i] = INFINITY;
        }
        // PrunedDTW
        if (sc <= min_ci) {} else {
            for (; ci<sc; ci++) {
                wps[ri_width + wpsi] = INFINITY;
                wpsi++;
            }
        }
        smaller_found = false;
        ec_next = ri;
        for (; ci<l2; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += EDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            if (d > p.max_step) { wps[ri_width + wpsi] = INFINITY; wpsi++; continue;}
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
            wpsi++;
        }
        ec = ec_next;
        for (idx_t i=ri_width + wpsi; i<ri_width + p.width; i++) {
            wps[i] = INFINITY;
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
        seq_t mir_value = INFINITY;
        idx_t mir_rel = 0;
        seq_t mic_value = INFINITY;
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
        rvalue = INFINITY;
    }
    
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
    
    return rvalue;
}

/*!
 Expand the compact wps datastructure to a full (l1+1)*(l2+1) sized matrix.
 */
void dtw_expand_wps(seq_t *wps, seq_t *full,
                    idx_t l1, idx_t l2,
                    DTWSettings *settings) {
    DTWWps p = dtw_wps_parts(l1, l2, settings);
    
    idx_t ri, ci, min_ci, max_ci, wpsi, wpsi_start;
    idx_t fwidth = l2 + 1;
    
    for (idx_t i=0; i<(l1+1)*(l2+1); i++) {
        full[i] = INFINITY;
    }
    
    // Top row: ri = -1
    full[0] = wps[0];
    wpsi = 1;
    for (ci=0; ci<MIN(p.width - 1, l2); ci++) {
        full[wpsi] = wps[wpsi];
        wpsi++;
    }
    
    // A. Rows: 0 <= ri < min(overlap_left_ri, overlap_right_ri)
    min_ci = 0;
    max_ci = p.window + p.ldiffc; // ri < overlap_right_i
    for (ri=0; ri<p.ri1; ri++) {
        full[fwidth*(ri + 1)] = wps[p.width*(ri + 1)];
        wpsi = 1;
        for (ci=min_ci; ci<max_ci; ci++) {
            full[(ri+1)*fwidth + ci + 1] = wps[(ri+1)*p.width + wpsi];
            wpsi++;
        }
        max_ci++;
    }
    
    // B. Rows: min(overlap_left_ri, overlap_right_ri) <= ri < overlap_left_ri
    min_ci = 0;
    max_ci = l2; // ri >= overlap_right_i
    for (ri=p.ri1; ri<p.ri2; ri++) {
        full[fwidth*(ri + 1)] = wps[p.width*(ri + 1)];
        wpsi = 1;
        for (ci=min_ci; ci<max_ci; ci++) {
            full[(ri+1)*fwidth + ci + 1] = wps[(ri+1)*p.width + wpsi];
            wpsi++;
        }
    }
    
    // C. Rows: overlap_left_ri <= ri < MAX(parts.overlap_left_ri, parts.overlap_right_ri)
    min_ci = 1;
    max_ci = 1 + 2 * p.window - 1 + p.ldiff;
    for (ri=p.ri2; ri<p.ri3; ri++) {
        full[(ri+1)*fwidth + min_ci] = wps[(ri+1)*p.width + 0];
        wpsi = 1;
        for (ci=min_ci; ci<max_ci; ci++) {
            full[(ri+1)*fwidth + ci + 1] = wps[(ri+1)*p.width + wpsi];
            wpsi++;
        }
        min_ci++;
        max_ci++;
    }
    
    // D. Rows: MAX(overlap_left_ri, overlap_right_ri) < ri <= l1
    min_ci = p.ri3 + 1 - p.window - p.ldiff;
    wpsi_start = 2;
    if (p.ri2 == p.ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
    }
    for (ri=p.ri3; ri<l1; ri++) {
        wpsi = wpsi_start;
        for (ci=min_ci; ci<l2; ci++) {
            full[(ri+1)*fwidth + ci + 1] = wps[(ri+1)*p.width + wpsi];
            wpsi++;
        }
        min_ci++;
        wpsi_start++;
    }
}

/*!
Compute best path between two series.
 
 @param wps Array of length `(l1+1)*min(l2+1, abs(l1-l2) + 2*window-1)` with the warping paths.
 @param i1 Array of length l1+l2 to store the indices for the first sequence.
    Reverse ordered, last one is if i1 or i2 is zero.
 @param i2 Array of length l1+l2 to store the indices for the second sequence.
    Reverse ordered, last one is if i1 or i2 is zero.
 @param l1 Length of first array.
 @param l2 Length of second array.
 @param settings for Dynamic Time Warping.
 @return length of path
 */
idx_t dtw_best_path(seq_t *wps, idx_t *i1, idx_t *i2, idx_t l1, idx_t l2, DTWSettings *settings) {
    DTWWps p = dtw_wps_parts(l1, l2, settings);
    
    idx_t i = 0;
    idx_t rip = l1;
    idx_t cip = l2;
    idx_t ri_widthp = p.width * (l1 - 1);
    idx_t ri_width = p.width * l1;
    idx_t min_ci;
    idx_t wpsi_start, wpsi;
    
    // D. ri3 <= ri < l1
    min_ci = p.ri3 + 1 - p.window - p.ldiff;
    wpsi_start = 2;
    if (p.ri2 == p.ri3) {
        wpsi_start = min_ci + 1;
    }
    wpsi = wpsi_start + (l2 - min_ci) - 1;
    while (rip > p.ri3 && cip > 0) {
        if (wps[ri_width + wpsi] != -1) {
            i1[i] = rip - 1;
            i2[i] = cip - 1;
            i++;
        }
        if (wps[ri_widthp + wpsi - 1] <= wps[ri_width  + wpsi - 1] &&
            wps[ri_widthp + wpsi - 1] <= wps[ri_widthp + wpsi]) {
            // Go diagonal
            cip--;
            rip--;
            wpsi--;
            ri_width = ri_widthp;
            ri_widthp -= p.width;
        } else if (wps[ri_width + wpsi - 1] <= wps[ri_widthp + wpsi]) {
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
        if (wps[ri_widthp + wpsi] <= wps[ri_width  + wpsi - 1] &&
            wps[ri_widthp + wpsi] <= wps[ri_widthp + wpsi + 1]) {
            // Go diagonal
            cip--;
            rip--;
            ri_width = ri_widthp;
            ri_widthp -= p.width;
        } else if (wps[ri_width + wpsi - 1] <= wps[ri_widthp + wpsi + 1]) {
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
        if (wps[ri_widthp + wpsi - 1] <= wps[ri_width  + wpsi - 1] &&
            wps[ri_widthp + wpsi - 1] <= wps[ri_widthp + wpsi]) {
            // Go diagonal
            cip--;
            rip--;
            wpsi--;
            ri_width = ri_widthp;
            ri_widthp -= p.width;
        } else {
            if (wps[ri_width + wpsi - 1] <= wps[ri_widthp + wpsi]) {
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
        wpsi_start = min_ci + 1;
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
 
 @return length of path
 */
idx_t warping_path(seq_t *from_s, idx_t from_l, seq_t* to_s, idx_t to_l, idx_t *from_i, idx_t *to_i, DTWSettings * settings) {
    return warping_path_ndim(from_s, from_l, to_s, to_l, from_i, to_i, 1, settings);
}

idx_t warping_path_ndim(seq_t *from_s, idx_t from_l, seq_t* to_s, idx_t to_l, idx_t *from_i, idx_t *to_i, int ndim, DTWSettings * settings) {
    idx_t path_length;
    idx_t wps_length = dtw_settings_wps_length(from_l, to_l, settings);
    seq_t *wps = (seq_t *)malloc(wps_length * sizeof(seq_t));
    dtw_warping_paths_ndim(wps, from_s, from_l, to_s, to_l, false, false, true,                        ndim, settings);
    path_length = dtw_best_path(wps, from_i, to_i, from_l, to_l, settings);
    free(wps);
    return path_length;
}

/*!
 Sample probabilistically warping path between two sequences.
 
 @return length of path
 */
idx_t warping_path_prob_ndim(seq_t *from_s, idx_t from_l, seq_t* to_s, idx_t to_l, idx_t *from_i, idx_t *to_i, seq_t avg, int ndim, DTWSettings * settings) {
    idx_t path_length;
    idx_t wps_length = dtw_settings_wps_length(from_l, to_l, settings);
    seq_t *wps = (seq_t *)malloc(wps_length * sizeof(seq_t));
    dtw_warping_paths_ndim(wps, from_s, from_l, to_s, to_l, false, false, true, ndim, settings);
    path_length = dtw_best_path_prob(wps, from_i, to_i, from_l, to_l, avg, settings);
    free(wps);
    return path_length;
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
 Keogh lower bound for DTW.
 */
seq_t lb_keogh(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, DTWSettings *settings) {
    idx_t window = settings->window;
    if (window == 0) {
        window = MAX(l1, l2);
    }
    idx_t imin, imax;
    idx_t t = 0;
    seq_t ui;
    seq_t li;
    seq_t ci;
    idx_t ldiff12 = l1 + 1;
    if (ldiff12 > l2) {
        ldiff12 -= l2;
        if (ldiff12 > window) {
            ldiff12 -= window;
        } else {
            ldiff12 = 0;
        }
    } else {
        ldiff12 = 0;
    }
    idx_t ldiff21 = l2 + window;
    if (ldiff21 > l1) {
        ldiff21 -= l1;
    } else {
        ldiff21 = 0;
    }
    
    for (idx_t i=0; i<l1; i++) {
        if (i > ldiff12) {
            imin = i - ldiff12;
        } else {
            imin = 0;
        }
        imax = MAX(l2, ldiff21);
        ui = 0;
        for (idx_t j=imin; j<imax; j++) {
            if (s2[j] > ui) {
                ui = s2[j];
            }
        }
        li = INFINITY;
        for (idx_t j=imin; j<imax; j++) {
            if (s2[j] < li) {
                li = s2[j];
            }
        }
        ci = s1[i];
        if (ci > ui) {
            t += ci - ui;
        } else if (ci < li) {
            t += li - ci;
        }
    }
    return t;
}


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


bool dtw_block_is_valid(DTWBlock *block, idx_t nb_series) {
    if (block->rb >= block->re) {
        printf("ERROR: Block row range is 0 or smaller\n");
        return false;
    }
    if (block->cb >= block->ce) {
        printf("ERROR: Block row range is 0 or smaller\n");
        return false;
    }
    if (block->rb >= nb_series) {
        printf("ERROR: Block rb exceeds number of series\n");
        return false;
    }
    if (block->re > nb_series) {
        printf("ERROR: Block re exceeds number of series\n");
        return false;
    }
    if (block->cb >= nb_series) {
        printf("ERROR: Block cb exceeds number of series\n");
        return false;
    }
    if (block->ce > nb_series) {
        printf("ERROR: Block ce exceeds number of series\n");
        return false;
    }
    return true;
}


// MARK: Distance Matrix


/*!
Distance matrix for n-dimensional DTW, executed on a list of pointers to arrays.

@param ptrs Pointers to arrays.  The arrays are expected to be 1-dimensional.
@param nb_ptrs Length of ptrs array
@param lengths Array of length nb_ptrs with all lengths of the arrays in ptrs.
@param output Array to store all outputs (should be (nb_ptrs-1)*nb_ptrs/2 if no block is given)
@param block Restrict to a certain block of combinations of series.
@param settings DTW settings
*/
idx_t dtw_distances_ptrs(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths, seq_t* output,
                          DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, cb;
    idx_t length;
    idx_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_ptrs);
    if (length == 0) {
        return 0;
    }
    
    // Correct block
    if (block->re == 0) {
        block->re = nb_ptrs;
    }
    if (block->ce == 0) {
        block->ce = nb_ptrs;
    }

    i = 0;
    for (r=block->rb; r<block->re; r++) {
        if (block->triu && r + 1 > block->cb) {
            cb = r+1;
        } else {
            cb = block->cb;
        }
        for (c=cb; c<block->ce; c++) {
            value = dtw_distance(ptrs[r], lengths[r],
                                 ptrs[c], lengths[c], settings);
//            printf("i=%zu - r=%zu - c=%zu - value=%.4f\n", i, r, c, value);
            output[i] = value;
            i += 1;
        }
    }
    return length;
}

/*!
Distance matrix for n-dimensional DTW, executed on a 2-dimensional array.
 
 The array is assumed to be C contiguous: C contiguous means that the array data is continuous in memory (see below) and that neighboring elements in the first dimension of the array are furthest apart in memory, whereas neighboring elements in the last dimension are closest together (from https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#brief-recap-on-c-fortran-and-strided-memory-layouts).

@param matrix 2-dimensional array. The order is defined by 1st dimension are the series, the 2nd dimension are the sequence entries.
@param nb_rows Number of series, size of the 1st dimension of matrix
@param nb_cols Number of elements in each series, size of the 2nd dimension of matrix
@param output Array to store all outputs (should be (nb_ptrs-1)*nb_ptrs/2 if no block is given)
@param block Restrict to a certain block of combinations of series.
@param settings DTW settings
*/
idx_t dtw_distances_matrix(seq_t *matrix, idx_t nb_rows, idx_t nb_cols, seq_t* output,
                           DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, cb;
    idx_t length;
    idx_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_rows);
    if (length == 0) {
        return 0;
    }
    
    // Correct block
    if (block->re == 0) {
        block->re = nb_rows;
    }
    if (block->ce == 0) {
        block->ce = nb_rows;
    }
    
    i = 0;
    for (r=block->rb; r<block->re; r++) {
        if (block->triu && r + 1 > block->cb) {
            cb = r+1;
        } else {
            cb = block->cb;
        }
        for (c=cb; c<block->ce; c++) {
            value = dtw_distance(&matrix[r*nb_cols], nb_cols,
                                 &matrix[c*nb_cols], nb_cols, settings);
//            printf("i=%zu - r=%zu - c=%zu - value=%.4f\n", i, r, c, value);
            output[i] = value;
            i += 1;
        }
    }
    assert(length == i);
    return length;
}

/*!
Distance matrix for n-dimensional DTW, executed on a 3-dimensional array and in parallel.
 
 The array is assumed to be C contiguous: C contiguous means that the array data is continuous in memory (see below) and that neighboring elements in the first dimension of the array are furthest apart in memory, whereas neighboring elements in the last dimension are closest together (from https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#brief-recap-on-c-fortran-and-strided-memory-layouts).

@param matrix 3-dimensional array. The order is defined by 1st dimension are the series, the 2nd dimension are the sequence entries, and the 3rd dimension are the n-dimensional values.
@param nb_rows Number of series, size of the 1st dimension of matrix
@param nb_cols Number of elements in each series, size of the 2nd dimension of matrix
@param ndim The number of dimensions in each sequence entry, size of the 3rd dimension of matrix
@param output Array to store all outputs (should be (nb_ptrs-1)*nb_ptrs/2 if no block is given)
@param block Restrict to a certain block of combinations of series.
@param settings DTW settings
*/
idx_t dtw_distances_ndim_matrix(seq_t *matrix, idx_t nb_rows, idx_t nb_cols, int ndim, seq_t* output,
                                 DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, cb;
    idx_t length;
    idx_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_rows);
    if (length == 0) {
        return 0;
    }
    
    // Correct block
    if (block->re == 0) {
        block->re = nb_rows;
    }
    if (block->ce == 0) {
        block->ce = nb_rows;
    }
    
    i = 0;
    for (r=block->rb; r<block->re; r++) {
        if (block->triu && r + 1 > block->cb) {
            cb = r+1;
        } else {
            cb = block->cb;
        }
        for (c=cb; c<block->ce; c++) {
            value = dtw_distance_ndim(&matrix[r*nb_cols*ndim], nb_cols,
                                      &matrix[c*nb_cols*ndim], nb_cols,
                                      ndim, settings);
//            printf("i=%zu - r=%zu - c=%zu - value=%.4f\n", i, r, c, value);
            output[i] = value;
            i += 1;
        }
    }
    assert(length == i);
    return length;
}


/*!
Distance matrix for n-dimensional DTW, executed on a list of pointers to arrays.

The arrays are assumed to be C contiguous: C contiguous means that the array data is continuous in memory (see below) and that neighboring elements in the first dimension of the array are furthest apart in memory, whereas neighboring elements in the last dimension are closest together (from https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#brief-recap-on-c-fortran-and-strided-memory-layouts).

@param ptrs Pointers to arrays. The order is defined by 1st dim is sequence entry, 2nd dim are the n-dimensional values. Thus the values for each n-dimensional entry are next to each other in the memory layout of the array.
@param nb_ptrs Length of ptrs array
@param lengths Array of length nb_ptrs with all lengths of the arrays in ptrs.
@param ndim The number of dimensions in each sequence entry
@param output Array to store all outputs (should be (nb_ptrs-1)*nb_ptrs/2 if no block is given)
@param block Restrict to a certain block of combinations of series.
@param settings DTW settings
*/
idx_t dtw_distances_ndim_ptrs(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths, int ndim, seq_t* output,
                               DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, cb;
    idx_t length;
    idx_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_ptrs);
    if (length == 0) {
        return 0;
    }
    
    // Correct block
    if (block->re == 0) {
        block->re = nb_ptrs;
    }
    if (block->ce == 0) {
        block->ce = nb_ptrs;
    }

    i = 0;
    for (r=block->rb; r<block->re; r++) {
        if (block->triu && r + 1 > block->cb) {
            cb = r+1;
        } else {
            cb = block->cb;
        }
        for (c=cb; c<block->ce; c++) {
            value = dtw_distance_ndim(ptrs[r], lengths[r],
                                      ptrs[c], lengths[c],
                                      ndim, settings);
//            printf("i=%zu - r=%zu - c=%zu - value=%.4f\n", i, r, c, value);
            output[i] = value;
            i += 1;
        }
    }
    return length;
}

idx_t dtw_distances_length(DTWBlock *block, idx_t nb_series) {
    // Note: int is usually 32-bit even on 64-bit systems
    idx_t ir;
    idx_t length = 0;  // Should be sidx_t but not available on all platforms
    idx_t overflow_buffer, delta;
    idx_t max_nb_series;
    
    if (block == NULL || block->re == 0 || block->ce == 0) {
        // Check for overflow
        max_nb_series = (idx_t) floor(sqrt(idx_t_max));
        if (nb_series > max_nb_series) {
            printf("ERROR: Length of array needed to represent the distance matrix for %zu series is larger than the maximal value allowed (unsigned %zu)\n", nb_series, idx_t_max);
            return 0;
        }
        if (block->triu) {
            // First divide the even number to avoid overflowing
            if (nb_series % 2 == 0) {
                length = (nb_series / 2) * (nb_series - 1);
            } else {
                length = nb_series * ((nb_series - 1) / 2);
            }
        } else { // triu=false
            length = nb_series * nb_series;
        }
    } else {
        if (!dtw_block_is_valid(block, nb_series)) {
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
                    printf("ERROR: Length of array needed to represent the distance matrix for %zu series and block {%zu, %zu, %zu, %zu} is larger than the maximal value allowed (unsigned %zu)\n", nb_series, block->rb, block->re, block->cb, block->ce, idx_t_max);
                    return 0;
                }
                length += delta;
            }
        } else { // triu=false
            // Check for overflow
            max_nb_series = idx_t_max / (block->re - block->rb);
            if ((block->ce - block->cb) > max_nb_series) {
                printf("ERROR: Length of array needed to represent the distance matrix for %zu series ", nb_series);
                printf("(in block %zd x %zd) is larger than the maximal value allowed (unsigned %zd)\n",
                        (block->re - block->rb), (block->ce - block->cb), idx_t_max);
                return 0;
            }
            length = (block->re - block->rb) * (block->ce - block->cb);
        }
    }
    return length;
}

// MARK: DBA

/*!
 Barycenter.
 
 Based on:
 F. Petitjean, A. Ketterlin, and P. Gan ̧carski.
 A global averaging method for dynamic time warping, with applications to clustering.
 Pattern Recognition, 44(3):678–693, 2011.
 
 @param ptrs Pointers to arrays.  The arrays are expected to be 1-dimensional.
 @param nb_ptrs Length of ptrs array
 @param lengths Array of length nb_ptrs with all lengths of the arrays in ptrs.
 @param c Initial average, afterwards the updated average
 @param t Length of average (typically this is the same as nb_cols)
            Real length is t*ndim.
 @param mask Bit-array
 @param prob_samples Probabilistically sample the best path samples number of times.
        Uses deterministic best path if samples is 0.
 @param settings Settings for distance functions
 */
void dtw_dba_ptrs(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths,
                  seq_t *c, idx_t t, ba_t *mask, int prob_samples, int ndim,
                  DTWSettings *settings) {
    seq_t *assoctab = (seq_t *)malloc(t * ndim * sizeof(seq_t));
    idx_t *assoctab_cnt = (idx_t *)malloc(t * sizeof(idx_t));
    idx_t r_idx = 0;
    idx_t max_length = 0;
    for (r_idx=0; r_idx<nb_ptrs; r_idx++) {
        if (lengths[r_idx] > max_length) {
            max_length = lengths[r_idx];
        }
    }

    idx_t *ci = (idx_t *)malloc((max_length + t) * sizeof(idx_t));
    idx_t *mi = (idx_t *)malloc((max_length + t) * sizeof(idx_t));
    idx_t pi, di;
    seq_t *sequence;
    seq_t *wps;
    seq_t avg_step;
    idx_t path_length;
    
    idx_t wps_length = dtw_settings_wps_length(t, max_length, settings);
    wps = (seq_t *)malloc(wps_length * sizeof(seq_t));
    
    for (pi=0; pi<t; pi++) {
        for (di=0; di<ndim; di++) {
            assoctab[pi * ndim + di] = 0;
        }
        assoctab_cnt[pi] = 0;
    }
    if (prob_samples == 0) {
        for (idx_t r=0; r<nb_ptrs; r++) {
            sequence = ptrs[r];
            if (bit_test(mask, r)) {
                // warping_path(c, t, sequence, lengths[r], ci, mi, settings);
                dtw_warping_paths_ndim(wps, c, t, sequence, lengths[r], false, false, true, ndim, settings);
                path_length = dtw_best_path(wps, ci, mi, t, lengths[r], settings);
                for (pi=0; pi<path_length; pi++) {
                    for (di=0; di<ndim; di++) {
                        assoctab[ci[pi]*ndim+di] += sequence[mi[pi]*ndim+di];
                    }
                    assoctab_cnt[ci[pi]] += 1;
                }
            }
        }
    } else {
        for (idx_t r=0; r<nb_ptrs; r++) {
            sequence = ptrs[r];
            if (bit_test(mask, r)) {
                avg_step = dtw_warping_paths_ndim(wps, c, t, sequence, lengths[r], true, false, true, ndim, settings);
                avg_step /= t;
                for (idx_t i_sample=0; i_sample<prob_samples; i_sample++) {
                    path_length = dtw_best_path_prob(wps, ci, mi, t, lengths[r], avg_step, settings);
                    for (pi=0; pi<path_length; pi++) {
                        for (di=0; di<ndim; di++) {
                            assoctab[ci[pi]*ndim+di] += sequence[mi[pi]*ndim+di];
                        }
                        assoctab_cnt[ci[pi]] += 1;
                    }
                }
            }
        }
    }
    for (idx_t i=0; i<t; i++) {
        if (assoctab_cnt[i] != 0) {
            for (di=0; di<ndim; di++) {
                c[i*ndim+di] = assoctab[i*ndim+di] / assoctab_cnt[i];
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

/*!
 Barycenter.
 
 Based on:
 F. Petitjean, A. Ketterlin, and P. Gan ̧carski.
 A global averaging method for dynamic time warping, with applications to clustering.
 Pattern Recognition, 44(3):678–693, 2011.
 
 @param matrix Sequences ordered in a matrix
 @param nb_rows Number of rows
 @param nb_cols Number of columns
 @param c Initial average, afterwards the updated average
 @param t Length of average (typically this is the same as nb_cols)
 @param mask Bit-array
 @param prob_samples Probabilistically sample the best path, sample number of times.
        Uses deterministic best path if samples is 0.
 @param settings Settings for distance functions
 */
void dtw_dba_matrix(seq_t *matrix, idx_t nb_rows, idx_t nb_cols,
                    seq_t *c, idx_t t, ba_t *mask, int prob_samples, int ndim,
                    DTWSettings *settings) {
    seq_t *assoctab = (seq_t *)malloc(t * ndim * sizeof(seq_t));
    idx_t *assoctab_cnt = (idx_t *)malloc(t * sizeof(idx_t));
    idx_t r_idx = 0;
    idx_t *ci = (idx_t *)malloc((nb_cols + t) * sizeof(idx_t));
    idx_t *mi = (idx_t *)malloc((nb_cols + t) * sizeof(idx_t));
    idx_t pi, di;
    seq_t *sequence;
    seq_t *wps;
    seq_t avg_step;
    idx_t path_length;
    
    idx_t wps_length = dtw_settings_wps_length(t, nb_cols, settings);
    wps = (seq_t *)malloc(wps_length * sizeof(seq_t));
    
    for (pi=0; pi<t; pi++) {
        for (di=0; di<ndim; di++) {
            assoctab[pi*ndim+di] = 0;
        }
        assoctab_cnt[pi] = 0;
    }
    if (prob_samples == 0) {
        for (idx_t r=0; r<nb_rows; r++) {
            sequence = &matrix[r_idx];
            if (bit_test(mask, r)) {
                dtw_warping_paths_ndim(wps, c, t, sequence, nb_cols, false, false, true, ndim, settings);
                path_length = dtw_best_path(wps, ci, mi, t, nb_cols, settings);
//                printf("best_path(%zu/%zu) = [", r+1, nb_rows);
//                for (idx_t i=0; i<path_length; i++) {
//                    printf(" %zu:(%zu,%zu)", i, ci[i], mi[i]);
//                }
//                printf("]\n");
                for (pi=0; pi<path_length; pi++) {
                    for (di=0; di<ndim; di++) {
                        assoctab[ci[pi]*ndim+di] += sequence[mi[pi]*ndim+di];
                    }
                    assoctab_cnt[ci[pi]] += 1;
//                    printf("[%zu] = [%zu] += %f\n", ci[pi], mi[pi], sequence[mi[pi]]);
                }
            }
            r_idx += nb_cols*ndim;
        }
    } else {
        for (idx_t r=0; r<nb_rows; r++) {
            sequence = &matrix[r_idx];
            if (bit_test(mask, r)) {
                avg_step = dtw_warping_paths_ndim(wps, c, t, sequence, nb_cols, true, false, true, ndim, settings);
                avg_step /= t;
                for (idx_t i_sample=0; i_sample<prob_samples; i_sample++) {
                    path_length = dtw_best_path_prob(wps, ci, mi, t, nb_cols, avg_step, settings);
//                    printf("best_path_prob = [");
//                    for (idx_t i=0; i<path_length; i++) {
//                        printf("(%zu,%zu)", ci[i], mi[i]);
//                    }
//                    printf("]\n");
                    for (pi=0; pi<path_length; pi++) {
                        for (di=0; di<ndim; di++) {
                            assoctab[ci[pi]*ndim+di] += sequence[mi[pi]*ndim+di];
                        }
                        assoctab_cnt[ci[pi]] += 1;
                    }
                }
            }
            r_idx += nb_cols*ndim;
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
    for (idx_t ri=0; ri<(l1+1); ri++) {
        for (idx_t wpsi=0; wpsi<p.width; wpsi++) {
            dtw_print_nb(wps[ri*p.width+wpsi]);
        }
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
        printf("],  # a\n");
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
        printf("],  # b\n");
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
        printf("],  # c\n");
        min_ci++;
        max_ci++;
    }
    
    // D. Rows: MAX(overlap_left_ri, overlap_right_ri) < ri <= l1
    min_ci = p.ri3 + 1 - p.window - p.ldiff;
    wpsi_start = 2;
    if (p.ri2 == p.ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
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
            printf("]]  # d\n");
        } else {
            printf("],  # d\n");
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
