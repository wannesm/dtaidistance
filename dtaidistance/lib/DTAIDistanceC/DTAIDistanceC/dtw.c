/*!
@file dtw.c
@brief DTAIDistance.dtw

@author Wannes Meert
@copyright Copyright © 2020 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/
#include "dtw.h"


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
        .use_ssize_t = false
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
dtwvalue dtw_distance(dtwvalue *s1, size_t l1,
                      dtwvalue *s2, size_t l2,
                      DTWSettings *settings) {
    size_t ldiff;
    size_t dl;
    signal(SIGINT, dtw_int_handler);
    
    size_t window = settings->window;
    dtwvalue max_step = settings->max_step;
    dtwvalue max_dist = settings->max_dist;
    dtwvalue penalty = settings->penalty;
    
    #ifdef DTWDEBUG
    printf("r=%zu, c=%zu\n", l1, l2);
    #endif
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
    if (max_dist == 0) {
        max_dist = INFINITY;
    } else {
        max_dist = pow(max_dist, 2);
    }
    penalty = pow(penalty, 2);
    size_t length = MIN(l2+1, ldiff + 2*window + 1);
    dtwvalue * dtw = (dtwvalue *)malloc(sizeof(dtwvalue) * length * 2);
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
    dtwvalue last_under_max_dist = 0;
    dtwvalue prev_last_under_max_dist = INFINITY;
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
    dtwvalue minv;
    dtwvalue d; // DTYPE_t
    dtwvalue tempv;
    dtwvalue psi_shortest = INFINITY;
    keepRunning = 1;
    for (i=0; i<l1; i++) {
        if (!keepRunning){
            printf("Stop computing DTW...\n");
            return INFINITY;
        }
        if (last_under_max_dist == -1) {
            prev_last_under_max_dist = INFINITY;
        } else {
            prev_last_under_max_dist = last_under_max_dist;
        }
        last_under_max_dist = -1;
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
            d = pow(s1[i] - s2[j], 2);
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
            if (dtw[curidx] <= max_dist) {
                last_under_max_dist = j;
            } else {
                dtw[curidx] = INFINITY;
                if (prev_last_under_max_dist + 1 - skipp < j + 1 - skip) {
                    break;
                }
            }
        }
        if (last_under_max_dist == -1) {
            #ifdef DTWDEBUG
            printf("early stop\n");
            dtw_print_twoline(dtw, l1, l2, length, i0, i1, skip, skipp, maxj, minj);
            #endif
            free(dtw);
            return INFINITY;
        }
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
    dtwvalue result = sqrt(dtw[length * i1 + l2 - skip]);
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
dtwvalue dtw_warping_paths(dtwvalue *wps,
                         dtwvalue *s1, size_t l1,
                         dtwvalue *s2, size_t l2,
                         bool return_dtw, bool do_sqrt,
                         DTWSettings *settings) {
    size_t ldiff;
    dtwvalue rvalue = 1;
    signal(SIGINT, dtw_int_handler);
    
    size_t window = settings->window;
    dtwvalue max_step = settings->max_step;
    dtwvalue max_dist = settings->max_dist;
    dtwvalue penalty = settings->penalty;
    
    #ifdef DTWDEBUG
    printf("r=%zu, c=%zu\n", l1, l2);
    #endif
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
    if (max_dist == 0) {
        max_dist = INFINITY;
    } else {
        max_dist = pow(max_dist, 2);
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
    dtwvalue last_under_max_dist = 0;
    dtwvalue prev_last_under_max_dist = INFINITY;
    size_t i0 = 1;
    size_t i1 = 0;
    size_t minj;
    size_t maxj;
    size_t dl;
    dtwvalue minv;
    dtwvalue d;
    dtwvalue tempv;
    keepRunning = 1;
    for (i=0; i<l1; i++) {
        if (!keepRunning){
            printf("Stop computing DTW...\n");
            return INFINITY;
        }
        if (last_under_max_dist == -1) {
            prev_last_under_max_dist = INFINITY;
        } else {
            prev_last_under_max_dist = last_under_max_dist;
        }
        last_under_max_dist = -1;
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
        for (j=maxj; j<minj; j++) {
            #ifdef DTWDEBUG
            printf("ri=%zu,ci=%zu, s1[i] = s1[%zu] = %f , s2[j] = s2[%zu] = %f\n", i, j, i, s1[i], j, s2[j]);
            #endif
            d = pow(s1[i] - s2[j], 2);
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
            printf("d = %f, minv = %f\n", d, minv);
            #endif
            wps[i1 * (l2 + 1) + j + 1] = d + minv;
            if (wps[i1 * (l2 + 1) + j + 1] <= max_dist) {
                last_under_max_dist = j;
            } else {
                wps[i1 * (l2 + 1) + j + 1] = INFINITY;
                if (prev_last_under_max_dist + 1 < j + 1) {
                    break;
                }
            }
        }
        if (last_under_max_dist == -1) {
            #ifdef DTWDEBUG
            printf("early stop\n");
            #endif
            if (do_sqrt) {
                for (i=0; i<(l1 + 1) * (l2 + 1); i++) {
                    wps[i] = sqrt(wps[i]);
                }
            }
            #ifdef DTWDEBUG
            dtw_print_wps(wps, l1, l2);
            #endif
            if (return_dtw) {
                return INFINITY;
            } else {
                return -1;
            }
        }
    }

    if (do_sqrt) {
        for (i=0; i<(l1 + 1) * (l2 + 1); i++) {
            wps[i] = sqrt(wps[i]);
        }
    }
    
    if (return_dtw && settings->psi == 0) {
        rvalue = wps[l1*(l2 + 1) + MIN(l2, l2 + window - 1)];
    } else if (return_dtw) {
        dtwvalue mir_value = INFINITY;
        size_t curi;
        size_t mir = 0;
        size_t mir_rel = 0;
        dtwvalue mic_value = INFINITY;
        size_t mic = 0;
        // Find smallest value in last column
        for (size_t ri=l1 - settings->psi; ri<l1; ri++) {
            curi = ri*(l2 + 1) + l2;
            if (wps[curi] < mir_value) {
                mir_value = wps[curi];
                mir = curi;
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
    
    return rvalue;
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


size_t dtw_distances_ptrs(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths, dtwvalue* output,
                          DTWBlock* block, DTWSettings* settings) {
    size_t r, c, cb;
    size_t length;
    size_t i;
    dtwvalue value;
    
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

size_t dtw_distances_matrix(dtwvalue *matrix, size_t nb_rows, size_t nb_cols, dtwvalue* output,
                           DTWBlock* block, DTWSettings* settings) {
    size_t r, c, cb;
    size_t length;
    size_t i;
    dtwvalue value;
    
    length = dtw_distances_length(block, nb_rows, settings->use_ssize_t);
    if (length == 0) {
        return 0;
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
    
//    printf("nb_series = %zu\n", nb_series);
//    dtw_print_block(block);
    
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
//                printf("1   - ir < block->cb = %zu\n", block->cb);
                delta = block->ce - block->cb;
            } else { // ir >= block->cb
//                printf("2   - ir >= block->cb = %zu\n", block->cb);
                if (block->ce <= ir) {
//                    printf("2.1 - block->ce = %zu <= ir\n", block->ce);
                    // ir only increases so block->ce will always be < ir
                    // delta = 0
                    break;
                } else { // block->ce > ir
//                    printf("2.2 - block->ce = %zu > ir\n", block->ce);
                    delta = block->ce - ir - 1;
                }
            }
            overflow_buffer = max_value - length;
            if (overflow_buffer < delta) {
                printf("Trying to execute %zu + %zu > %zu\n", length, delta, max_value);
                printf("ERROR: Length of array needed to represent the distance matrix for %zu series and block {%zu, %zu, %zu, %zu} is larger than the maximal value allowed (unsigned %zu)\n", nb_series, block->rb, block->re, block->cb, block->ce, max_value);
                return 0;
            }
//            printf("ir = %zu // delta = %zu\n", ir, delta);
            length += delta;
        }
    }
//    printf("length = %zu\n", length);
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
void dtw_print_wps(dtwvalue * wps, size_t l1, size_t l2) {
    char buffer[20];
    char format[5];
    snprintf(format, sizeof(format), "%%.%df", printPrecision);
    for (int ri=0; ri<l1+1; ri++) {
        if (ri==0) {
            printf("[[ ");
        } else {
            printf(" [ ");
        }
        for (int ci=0; ci<l1+1; ci++) {
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
void dtw_print_twoline(dtwvalue * dtw, size_t r, size_t c, size_t length, int i0, int i1, size_t skip, size_t skipp, size_t maxj, size_t minj) {
    char buffer[20];
    char format[5];
    snprintf(format, sizeof(format), "%%.%df ", printPrecision);
    size_t ci_cor; // corrected column index
    // Row 1
    printf("[[ ");
    for (int ci=0; ci<c; ci++) {
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
    for (int ci=0; ci<c; ci++) {
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
