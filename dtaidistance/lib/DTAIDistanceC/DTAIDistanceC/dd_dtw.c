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
        .psi = 0,
        .use_ssize_t = false,
        .use_pruning = false,
        .only_ub = false
    };
    return s;
}

void dtw_settings_print(DTWSettings *settings) {
    printf("DTWSettings {\n");
    printf("  window = %zu\n", settings->window);
    printf("  max_dist = %f\n", settings->max_dist);
    printf("  max_step = %f\n", settings->max_step);
    printf("  max_length_diff = %zu\n", settings->max_length_diff);
    printf("  penalty = %f\n", settings->penalty);
    printf("  psi = %zu\n", settings->psi);
    printf("  use_ssize_t = %d\n", settings->use_ssize_t);
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
seq_t dtw_distance(seq_t *s1, size_t l1,
                      seq_t *s2, size_t l2,
                      DTWSettings *settings) {
    assert(settings->psi < l1 && settings->psi < l2);
    size_t ldiff;
    size_t dl;
    // DTWPruned
    size_t sc = 0;
    size_t ec = 0;
    bool smaller_found;
    size_t ec_next;
    signal(SIGINT, dtw_int_handler);
    
    size_t window = settings->window;
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
    size_t length = MIN(l2+1, ldiff + 2*window + 1);
    assert(length > 0);
    seq_t * dtw = (seq_t *)malloc(sizeof(seq_t) * length * 2);
    if (!dtw) {
        printf("Error: dtw_distance - Cannot allocate memory (size=%zu)\n", length*2);
        return 0;
    }
    size_t i;
    size_t j;
    for (j=0; j<length*2; j++) {
        dtw[j] = INFINITY;
    }
    for (i=0; i<settings->psi + 1; i++) {
        dtw[i] = 0;
    }
    size_t skip = 0;
    size_t skipp = 0;
    int i0 = 1;
    int i1 = 0;
    size_t minj;
    size_t maxj;
    size_t curidx;
    size_t dl_window = dl + window - 1;
    size_t ldiff_window = window;
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
        if (settings->psi != 0 && maxj == 0 && i < settings->psi) {
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
        if (settings->psi != 0 && minj == l2 && l1 - 1 - i <= settings->psi) {
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
    // Deal wit psi-relaxation
    if (settings->psi != 0) {
        for (i=l2 - skip - settings->psi; i<l2 - skip + 1; i++) { // iterate over vci
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

/**
 Compute the DTW between two n-dimensional series.

 @param s1 First sequence
 @param l1 Length of first sequence. In tuples, real length should be length*ndim.
 @param s2 Second sequence
 @param l2 Length of second sequence. In tuples, real length should be length*ndim.
 @param ndim Number of dimensions
 @param settings A DTWSettings struct with options for the DTW algorithm.
*/
seq_t dtw_distance_ndim(seq_t *s1, size_t l1,
                           seq_t *s2, size_t l2, int ndim,
                           DTWSettings *settings) {
    assert(settings->psi < l1 && settings->psi < l2);
    size_t ldiff;
    size_t dl;
    // DTWPruned
    size_t sc = 0;
    size_t ec = 0;
    bool smaller_found;
    size_t ec_next;
    signal(SIGINT, dtw_int_handler);
    
    size_t window = settings->window;
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
    size_t length = MIN(l2+1, ldiff + 2*window + 1);
    assert(length > 0);
    seq_t * dtw = (seq_t *)malloc(sizeof(seq_t) * length * 2);
    if (!dtw) {
        printf("Error: dtw_distance - Cannot allocate memory (size=%zu)\n", length*2);
        return 0;
    }
    size_t i;
    size_t j;
    size_t i_idx;
    size_t j_idx;
    for (j=0; j<length*2; j++) {
        dtw[j] = INFINITY;
    }
    for (i=0; i<settings->psi + 1; i++) {
        dtw[i] = 0;
    }
    size_t skip = 0;
    size_t skipp = 0;
    int i0 = 1;
    int i1 = 0;
    size_t minj;
    size_t maxj;
    size_t curidx;
    size_t dl_window = dl + window - 1;
    size_t ldiff_window = window;
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
        if (settings->psi != 0 && maxj == 0 && i < settings->psi) {
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
        if (settings->psi != 0 && minj == l2 && l1 - 1 - i <= settings->psi) {
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
    // Deal wit psi-relaxation
    if (settings->psi != 0) {
        for (i=l2 - skip - settings->psi; i<l2 - skip + 1; i++) { // iterate over vci
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


/*!
Compute all warping paths between two series.
 
@param wps Empty array of length `(l1+1)*(l2+1)` in which the warping paths will be stored.
    It represents the full matrix of warping paths between the two series.
@param s1 First sequence
@param l1 Length of first sequence
@param s2 Second sequence
@param l2 Length of second sequence
@param return_dtw If only the matrix is required, finding the dtw value can be skipped
    to save operations.
@param do_sqrt Apply the sqrt operations on all items in the wps array. If not required,
    this can be skipped to save operations.
@param settings A DTWSettings struct with options for the DTW algorithm.
 
@return The dtw value if return_dtw is true; Otherwise -1.
*/
seq_t dtw_warping_paths(seq_t *wps,
                         seq_t *s1, size_t l1,
                         seq_t *s2, size_t l2,
                         bool return_dtw, bool do_sqrt,
                         DTWSettings *settings) {
    size_t ldiff;
    // DTWPruned
    size_t sc = 0;
    size_t ec = 0;
    bool smaller_found;
    size_t ec_next;
    seq_t rvalue = 1;
    signal(SIGINT, dtw_int_handler);
    
    size_t window = settings->window;
    seq_t max_step = settings->max_step;
    seq_t max_dist = settings->max_dist; // upper bound
    seq_t penalty = settings->penalty;
    
    #ifdef DTWDEBUG
    printf("r=%zu, c=%zu\n", l1, l2);
    #endif
    if (settings->use_pruning) {
        max_dist = pow(ub_euclidean(s1, l1, s2, l2), 2);
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
    if (settings->max_length_diff != 0 && ldiff > settings->max_length_diff) {
        #ifdef DTWDEBUG
        printf("Early stop: max_length_diff");
        #endif
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
    size_t i;
    size_t j;
    for (j=0; j<(l1 + 1) * (l2 + 1); j++) {
        wps[j] = INFINITY;
    }
    for (i=0; i<settings->psi + 1; i++) {
        wps[i] = 0;
        wps[i * (l2 + 1)] = 0;
    }
    size_t i0 = 1;
    size_t i1 = 0;
    size_t minj;
    size_t maxj;
    size_t dl;
    seq_t minv;
    seq_t d;
    seq_t tempv;
    keepRunning = 1;
    for (i=0; i<l1; i++) {
        #ifdef DTWDEBUG
        printf("i=%zu, sc=%zu, ec=%zu\n", i, sc, ec);
        #endif
        if (!keepRunning){
            printf("Stop computing DTW...\n");
            return INFINITY;
        }
        i0 = i;
        i1 = i + 1;
        if (l1 > l2) {
            dl = ldiff;
        } else {
            dl = 0;
        }
        maxj = i + 1;
        if (maxj > dl) {
            maxj -= dl;
            if (maxj > window) {
                maxj -= window;
            } else {
                maxj = 0;
            }
        } else {
            maxj = 0;
        }
        minj = i + window;
        if (l2 > l1) {
            minj += ldiff;
        }
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
        for (j=maxj; j<minj; j++) {
            #ifdef DTWDEBUG
            printf("ri=%zu, ci=%zu, s1[i/%zu]=%f , s2[j/%zu]=%f\n", i, j, i, s1[i], j, s2[j]);
            #endif
            d = EDIST(s1[i], s2[j]);
            if (d > max_step) {
                continue;
            }
            minv = wps[i0 * (l2 + 1) + j];
            tempv = wps[i0 * (l2 + 1) + j + 1] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            tempv = wps[i1 * (l2 + 1) + j] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            #ifdef DTWDEBUG
            printf("wps[%zu,%zu]=%f, d=%f, minv =%f\n", i, j, d+minv, d, minv);
            #endif
            wps[i1 * (l2 + 1) + j + 1] = d + minv;
            // PrunedDTW
            if (wps[i1 * (l2 + 1) + j + 1] > max_dist) {
                if (!smaller_found) {
                    sc = j + 1;
                }
                if (j >= ec) {
                    break;
                }
            } else {
                smaller_found = true;
                ec_next = j + 1;
            }
        }
        ec = ec_next;
    }

    if (do_sqrt) {
        for (i=0; i<(l1 + 1) * (l2 + 1); i++) {
            wps[i] = sqrt(wps[i]);
        }
    }
    
    // Deal with Psi-relaxation
    if (return_dtw && settings->psi == 0) {
        rvalue = wps[l1*(l2 + 1) + MIN(l2, l2 + window - 1)];
    } else if (return_dtw) {
        seq_t mir_value = INFINITY;
        size_t curi;
        size_t mir_rel = 0;
        seq_t mic_value = INFINITY;
        size_t mic = 0;
        // Find smallest value in last column
        for (size_t ri=l1 - settings->psi; ri<l1; ri++) {
            curi = ri*(l2 + 1) + l2;
            if (wps[curi] < mir_value) {
                mir_value = wps[curi];
                mir_rel = ri;
            }
        }
        // Find smallest value in last row
        for (size_t ci=l2 - settings->psi; ci<l2; ci++) {
            curi = l1*(l2 + 1) + ci;
            if (wps[curi] < mic_value) {
                mic_value = wps[curi];
                mic = curi;
            }
        }
        // Set values with higher indices than the smallest value to -1
        // and return smallest value as DTW
        if (mir_value < mic_value) {
            for (size_t ri=mir_rel + 1; ri<l1 + 1; ri++) {
                curi = ri*(l2 + 1) + l2;
                wps[curi] = -1;
            }
            rvalue = mir_value;
        } else {
            for (curi=mic + 1; curi<(l1 + 1) * (l2 + 1); curi++) {
                wps[curi] = -1;
            }
            rvalue =  mic_value;
        }
    } else {
        rvalue = -1;
    }
    
    #ifdef DTWDEBUG
    dtw_print_wps(wps, l1, l2);
    #endif
    if (settings->max_dist > 0 && rvalue > settings->max_dist) {
        // DTWPruned keeps the last value larger than max_dist. Correct for this.
        rvalue = INFINITY;
    }
    return rvalue;
}


// MARK: Bounds

/*!
 Euclidean upper bound for DTW.
 
 @see ed.euclidean_distance.
 */
seq_t ub_euclidean(seq_t *s1, size_t l1, seq_t *s2, size_t l2) {
    return euclidean_distance(s1, l1, s2, l2);
}


/*!
 Euclidean upper bound for DTW.
 
 @see ed.euclidean_distance_ndim.
*/
seq_t ub_euclidean_ndim(seq_t *s1, size_t l1, seq_t *s2, size_t l2, int ndim) {
    return euclidean_distance_ndim(s1, l1, s2, l2, ndim);
}


/*!
 Keogh lower bound for DTW.
 */
seq_t lb_keogh(seq_t *s1, size_t l1, seq_t *s2, size_t l2, DTWSettings *settings) {
    size_t window = settings->window;
    if (window == 0) {
        window = MAX(l1, l2);
    }
    size_t imin, imax;
    size_t t = 0;
    seq_t ui;
    seq_t li;
    seq_t ci;
    size_t ldiff12 = l1 + 1;
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
    size_t ldiff21 = l2 + window;
    if (ldiff21 > l1) {
        ldiff21 -= l1;
    } else {
        ldiff21 = 0;
    }
    
    for (size_t i=0; i<l1; i++) {
        if (i > ldiff12) {
            imin = i - ldiff12;
        } else {
            imin = 0;
        }
        imax = MAX(l2, ldiff21);
        ui = 0;
        for (size_t j=imin; j<imax; j++) {
            if (s2[j] > ui) {
                ui = s2[j];
            }
        }
        li = INFINITY;
        for (size_t j=imin; j<imax; j++) {
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
        .ce = 0   // column-end
    };
    return b;
}


void dtw_block_print(DTWBlock *block) {
    printf("DTWBlock {\n");
    printf("  rb = %zu\n", block->rb);
    printf("  re = %zu\n", block->re);
    printf("  cb = %zu\n", block->cb);
    printf("  ce = %zu\n", block->ce);
    printf("}\n");
}


bool dtw_block_is_valid(DTWBlock *block, size_t nb_series) {
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
size_t dtw_distances_ptrs(seq_t **ptrs, size_t nb_ptrs, size_t* lengths, seq_t* output,
                          DTWBlock* block, DTWSettings* settings) {
    size_t r, c, cb;
    size_t length;
    size_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_ptrs, false);
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
        if (r + 1 > block->cb) {
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
size_t dtw_distances_matrix(seq_t *matrix, size_t nb_rows, size_t nb_cols, seq_t* output,
                           DTWBlock* block, DTWSettings* settings) {
    size_t r, c, cb;
    size_t length;
    size_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_rows, settings->use_ssize_t);
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
        if (r + 1 > block->cb) {
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
size_t dtw_distances_ndim_matrix(seq_t *matrix, size_t nb_rows, size_t nb_cols, int ndim, seq_t* output,
                                 DTWBlock* block, DTWSettings* settings) {
    size_t r, c, cb;
    size_t length;
    size_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_rows, settings->use_ssize_t);
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
        if (r + 1 > block->cb) {
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
size_t dtw_distances_ndim_ptrs(seq_t **ptrs, size_t nb_ptrs, size_t* lengths, int ndim, seq_t* output,
                               DTWBlock* block, DTWSettings* settings) {
    size_t r, c, cb;
    size_t length;
    size_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_ptrs, false);
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
        if (r + 1 > block->cb) {
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

size_t dtw_distances_length(DTWBlock *block, size_t nb_series, bool use_ssize_t) {
    // Note: int is usually 32-bit even on 64-bit systems
    size_t ir;
    size_t length = 0;  // Should be ssize_t but not available on all platforms
    size_t overflow_buffer, delta;
    size_t max_nb_series;
    size_t max_value;
    if (use_ssize_t) {
        max_value = PTRDIFF_MAX;
    } else {
        max_value = SIZE_MAX;
    }
    
    if (block->re == 0 || block->ce == 0) {
        // Check for overflow
        if (use_ssize_t) {
            max_nb_series = floor(sqrt(max_value));
        } else {
            max_nb_series = floor(sqrt(max_value));
        }
        if (nb_series > max_nb_series) {
            printf("ERROR: Length of array needed to represent the distance matrix for %zu series is larger than the maximal value allowed (unsigned %zu)\n", nb_series, max_value);
            return 0;
        }
        // First divide the even number to avoid overflowing
        if (nb_series % 2 == 0) {
            length = (nb_series / 2) * (nb_series - 1);
        } else {
            length = nb_series * ((nb_series - 1) / 2);
        }
    } else {
        if (!dtw_block_is_valid(block, nb_series)) {
            return 0;
        }
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
            overflow_buffer = max_value - length;
            if (overflow_buffer < delta) {
                printf("Trying to execute %zu + %zu > %zu\n", length, delta, max_value);
                printf("ERROR: Length of array needed to represent the distance matrix for %zu series and block {%zu, %zu, %zu, %zu} is larger than the maximal value allowed (unsigned %zu)\n", nb_series, block->rb, block->re, block->cb, block->ce, max_value);
                return 0;
            }
            length += delta;
        }
    }
    return length;
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
void dtw_print_wps(seq_t * wps, size_t l1, size_t l2) {
    char buffer[20];
    char format[5];
    snprintf(format, sizeof(format), "%%.%df", printPrecision);
    for (size_t ri=0; ri<l1+1; ri++) {
        if (ri==0) {
            printf("[[ ");
        } else {
            printf(" [ ");
        }
        for (size_t ci=0; ci<l1+1; ci++) {
            snprintf(buffer, sizeof(buffer), format, wps[ri*(l2+1)+ci]);
            printf("%-*s", printPrecision + 3, buffer);
        }
        if (ri==l1) {
            printf("]]\n");
        } else {
            printf("]\n");
        }
    }
}

/* Helper function for debugging. */
void dtw_print_twoline(seq_t * dtw, size_t r, size_t c, size_t length, int i0, int i1, size_t skip, size_t skipp, size_t maxj, size_t minj) {
    char buffer[20];
    char format[5];
    snprintf(format, sizeof(format), "%%.%df ", printPrecision);
    size_t ci_cor; // corrected column index
    // Row 1
    printf("[[ ");
    for (size_t ci=0; ci<c; ci++) {
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
    for (size_t ci=0; ci<c; ci++) {
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
