//
//  dtw.c
//  DTAIDistance
//
//  Copyright © 2020 Wannes Meert.
//  Apache License, Version 2.0, see LICENSE for details.
//

#include "dtw.h"


//#define DTWDEBUG


void intHandler(int dummy) {
    printf("Interrupt process, stopping ...\n");
    keepRunning = 0;
}

// MARK: DTW

/* Create settings struct with default values (all extras deactivated). */
DTWSettings dtw_default_settings(void) {
    DTWSettings s = {
        .window = 0,
        .max_dist = 0,
        .max_step = 0,
        .max_length_diff = 0,
        .penalty = 0,
        .psi = 0
    };
    return s;
}

/*
 Compute the DTW between two series.
 
 @param s1: First sequence
 @param l1: length of first sequence
 @param s2: Second sequence
 @param l2: length of second sequence
 @param settings: A struct with options for the DTW algorithm.
 */
dtwvalue dtw_distance(dtwvalue *s1, int l1,
                      dtwvalue *s2, int l2,
                      DTWSettings *settings) {
    signal(SIGINT, intHandler);
    
    int window = settings->window;
    dtwvalue max_step = settings->max_step;
    dtwvalue max_dist = settings->max_dist;
    dtwvalue penalty = settings->penalty;
    
    #ifdef DTWDEBUG
    printf("r=%i, c=%i\n", l1, l2);
    #endif
    if (settings->max_length_diff != 0 && abs(l1-l2) > settings->max_length_diff) {
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
    int length = MIN(l2+1, abs(l1-l2) + 2*window + 1);
    dtwvalue * dtw = (dtwvalue *)malloc(sizeof(dtwvalue) * length * 2);
    if (!dtw) {
        printf("Error: dtw_distance - Cannot allocate memory (size=%d)\n", length*2);
        return 0;
    }
    int i;
    int j;
    for (j=0; j<length*2; j++) {
        dtw[j] = INFINITY;
    }
    for (i=0; i<settings->psi + 1; i++) {
        dtw[i] = 0;
    }
    dtwvalue last_under_max_dist = 0;
    dtwvalue prev_last_under_max_dist = INFINITY;
    int skip = 0;
    int skipp = 0;
    int i0 = 1;
    int i1 = 0;
    int minj;
    int maxj;
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
        maxj = l1 - l2;
        if (maxj < 0) {
            maxj = 0;
        }
        maxj = i - maxj - window + 1;
        if (maxj < 0) {
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
        minj = l2 - l1;
        if (minj < 0) {
            minj = 0;
        }
        minj = i + minj + window;
        if (minj > l2) {
            minj = l2;
        }
        if (settings->psi != 0 && maxj == 0 && i < settings->psi) {
            dtw[i1*length + 0] = 0;
        }
        for (j=maxj; j<minj; j++) {
            #ifdef DTWDEBUG
            printf("ri=%i,ci=%i, s1[i] = s1[%i] = %f , s2[j] = s2[%i] = %f\n", i, j, i, s1[i], j, s2[j]);
            #endif
            d = pow(s1[i] - s2[j], 2);
            if (d > max_step) {
                continue;
            }
            minv = dtw[i0*length + j - skipp];
            tempv = dtw[i0*length + j + 1 - skipp] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            tempv = dtw[i1*length + j - skip] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            #ifdef DTWDEBUG
            printf("d = %f, minv = %f\n", d, minv);
            #endif
            dtw[i1 * length + j + 1 - skip] = d + minv;
            #ifdef DTWDEBUG
            printf("%i, %i, %i\n",i0*length + j - skipp,i0*length + j + 1 - skipp,i1*length + j - skip);
            printf("%f, %f, %f\n",dtw[i0*length + j - skipp],dtw[i0*length + j + 1 - skipp],dtw[i1*length + j - skip]);
            printf("i=%i, j=%i, d=%f, skip=%i, skipp=%i\n",i,j,d,skip,skipp);
            #endif
            if (dtw[i1*length + j + 1 - skip] <= max_dist) {
                last_under_max_dist = j;
            } else {
                dtw[i1*length + j + 1 - skip] = INFINITY;
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
            if (dtw[i1*length + length - 1] < psi_shortest) {
                psi_shortest = dtw[i1*length + length - 1];
            }
        }
#ifdef DTWDEBUG
        dtw_print_twoline(dtw, l1, l2, length, i0, i1, skip, skipp, maxj, minj);
#endif
    }
    if (window - 1 < 0) {
        l2 = l2 + window - 1;
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

/*
Compute all warping paths between two series.

@param wps: Empty array of length (l1+1)*(l2+1) in which the warping paths will be stored.
    It represents the full matrix of warping paths between the two series.
@param s1: First sequence
@param l1: length of first sequence
@param s2: Second sequence
@param l2: length of second sequence
@param return_dtw: If only the matrix is required, finding the dtw value can be skipped
    to save operations.
@param do_sqrt: Apply the sqrt operations on all items in the wps array. If not required,
    this can be skipped to save operations.
@param settings: A struct with options for the DTW algorithm.
 
@return The dtw value if return_dtw is true; Otherwise -1.
*/
dtwvalue dtw_warping_paths(dtwvalue *wps,
                         dtwvalue *s1, int l1,
                         dtwvalue *s2, int l2,
                         bool return_dtw, bool do_sqrt,
                         DTWSettings *settings) {
    dtwvalue rvalue = 1;
    signal(SIGINT, intHandler);
    
    int window = settings->window;
    dtwvalue max_step = settings->max_step;
    dtwvalue max_dist = settings->max_dist;
    dtwvalue penalty = settings->penalty;
    
    #ifdef DTWDEBUG
    printf("r=%i, c=%i\n", l1, l2);
    #endif
    if (settings->max_length_diff != 0 && abs(l1-l2) > settings->max_length_diff) {
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
    int i;
    int j;
    for (j=0; j<(l1 + 1) * (l2 + 1); j++) {
        wps[j] = INFINITY;
    }
    for (i=0; i<settings->psi + 1; i++) {
        wps[i] = 0;
        wps[i * (l2 + 1)] = 0;
    }
    dtwvalue last_under_max_dist = 0;
    dtwvalue prev_last_under_max_dist = INFINITY;
    int i0 = 1;
    int i1 = 0;
    int minj;
    int maxj;
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
        maxj = l1 - l2;
        if (maxj < 0) {
            maxj = 0;
        }
        maxj = i - maxj - window + 1;
        if (maxj < 0) {
            maxj = 0;
        }
        minj = l2 - l1;
        if (minj < 0) {
            minj = 0;
        }
        minj = i + minj + window;
        if (minj > l2) {
            minj = l2;
        }
        for (j=maxj; j<minj; j++) {
            #ifdef DTWDEBUG
            printf("ri=%i,ci=%i, s1[i] = s1[%i] = %f , s2[j] = s2[%i] = %f\n", i, j, i, s1[i], j, s2[j]);
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
        int curi;
        int mir = 0;
        int mir_rel = 0;
        dtwvalue mic_value = INFINITY;
        int mic = 0;
        // Find smallest value in last column
        for (int ri=l1 - settings->psi; ri<l1; ri++) {
            curi = ri*(l2 + 1) + l2;
            if (wps[curi] < mir_value) {
                mir_value = wps[curi];
                mir = curi;
                mir_rel = ri;
            }
        }
        // Find smallest value in last row
        for (int ci=l2 - settings->psi; ci<l2; ci++) {
            curi = l1*(l2 + 1) + ci;
            if (wps[curi] < mic_value) {
                mic_value = wps[curi];
                mic = curi;
            }
        }
        // Set values with higher indices than the smallest value to -1
        // and return smallest value as DTW
        if (mir_value < mic_value) {
            for (int ri=mir_rel + 1; ri<l1 + 1; ri++) {
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


// MARK: Distance Matrix

/* Create settings struct with default values (all extras deactivated). */
DTWBlock dtw_empty_block(void) {
    DTWBlock b = {
        .rb = 0,  // row-begin
        .re = 0,  // row-end
        .cb = 0,  // column-begin
        .ce = 0   // column-end
    };
    return b;
}


size_t dtw_distances_ptrs(dtwvalue **ptrs, int nb_ptrs, int* lengths, dtwvalue* output,
                          DTWBlock* block, DTWSettings* settings) {
    int r, c, cb;
    size_t length;
    size_t i;
    dtwvalue value;
    
    length = dtw_distances_length(block, nb_ptrs);
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
            value = dtw_distance(ptrs[r], lengths[r],
                                 ptrs[c], lengths[c], settings);
//            printf("i=%zu - r=%zu - c=%zu - value=%.4f\n", i, r, c, value);
            output[i] = value;
            i += 1;
        }
    }
    return length;
}

size_t dtw_distances_matrix(dtwvalue *matrix, int nb_rows, int nb_cols, dtwvalue* output,
                           DTWBlock* block, DTWSettings* settings) {
    int r, c, cb;
    size_t length;
    size_t i;
    dtwvalue value;
    
    length = dtw_distances_length(block, nb_rows);
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

size_t dtw_distances_length(DTWBlock *block, int nb_series) {
    size_t ir;
    long llength = 0;
    size_t slength;

    if (block->re == 0 && block->ce == 0) {
        // First divide the even number to avoid overflowing
        if (nb_series % 2 == 0) {
            llength = (nb_series / 2) * (nb_series - 1);
        } else {
            llength = nb_series * ((nb_series - 1) / 2);
        }
    } else {
        for (ir=block->rb; ir<block->re; ir++) {
            if (block->cb <= ir) {
                if (block->ce > ir) {
                    llength += (block->ce - ir - 1);
                }
            } else {
                if (block->ce > ir) {
                    llength += (block->ce - block->cb);
                }
            }
        }
    }
    slength = llength;
    #ifdef DTWDEBUG
    printf("length=%f, llength=%f", slength, llength);
    #endif
    if (slength < 0) {
        printf("ERROR: Length of array needed to represent the distance matrix larger than maximal value for size_t");
        return 0;
    }
    
    // Correct block
    if (block->re == 0) {
        block->re = nb_series;
    }
    if (block->ce == 0) {
        block->ce = nb_series;
    }
    
    return slength;
}


// MARK: Debugging

void dtw_set_printprecision(int precision) {
    printPrecision = precision;
}

void dtw_reset_printprecision(void) {
    printPrecision = 3;
}

/* Helper function for debugging. */
void dtw_print_wps(dtwvalue * wps, int l1, int l2) {
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
void dtw_print_twoline(dtwvalue * dtw, int r, int c, int length, int i0, int i1, int skip, int skipp, int maxj, int minj) {
    char buffer[20];
    char format[5];
    snprintf(format, sizeof(format), "%%.%df ", printPrecision);
    int ci_cor; // corrected column index
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

void dtw_print_settings(DTWSettings *settings) {
    printf("DTWSettings {\n");
    printf("  window = %d\n", settings->window);
    printf("  max_dist = %f\n", settings->max_dist);
    printf("  max_step = %f\n", settings->max_step);
    printf("  max_length_diff = %d\n", settings->max_length_diff);
    printf("  penalty = %f\n", settings->penalty);
    printf("  psi = %d\n", settings->psi);
    printf("}\n");
}
