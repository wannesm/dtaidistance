//
//  dtw_openmp.c
//  DTAIDistanceC
//
//  Copyright © 2020 Wannes Meert.
//  Apache License, Version 2.0, see LICENSE for details.
//

#include "dtw_openmp.h"


int dtw_distances_prepare(DTWBlock *block, int len_cur, ssize_t **irs, ssize_t **ics, size_t *length) {
    size_t r, c;
    size_t cb, ir;
    
    *length = dtw_distances_length(block, len_cur);
    if (length == 0) {
        return 0;
    }

    *irs = (ssize_t *)malloc(sizeof(ssize_t) * *length);
    if (!irs) {
        printf("Error: dtw_matrix_parallel - cannot allocate memory (length = %zu)", *length);
        return 1;
    }
    *ics = (ssize_t *)malloc(sizeof(ssize_t) * *length);
    if (!ics) {
        printf("Error: dtw_matrix_parallel - cannot allocate memory (length = %zu)", *length);
        return 1;
    }
    ir = 0;
    for (r=block->rb; r<block->re; r++) {
        if (r + 1 > block->cb) {
            cb = r+1;
        } else {
            cb = block->cb;
        }
        for (c=cb; c<block->ce; c++) {
            (*irs)[ir] = r;
            (*ics)[ir] = c;
            ir += 1;
        }
    }
    return 0;
}


size_t dtw_distances_ptrs_parallel(dtwvalue **ptrs, int nb_ptrs, int* lengths, dtwvalue* output,
                     DTWBlock* block, DTWSettings* settings) {
    // Requires openmp which is not supported for clang on mac by default (use newer version of clang)
    size_t r, c;
    size_t length;
    ssize_t *irs, *ics;

    if (dtw_distances_prepare(block, nb_ptrs, &irs, &ics, &length) != 0) {
        return 0;
    }

    size_t pi=0;
    #pragma omp parallel for private(pi, r, c)
    for(pi=0; pi<length; ++pi) {
//        printf("pi=%zu\n", pi);
        r = irs[pi];
        c = ics[pi];
        double value = dtw_distance(ptrs[r], lengths[r],
                                    ptrs[c], lengths[c], settings);
//        printf("pi=%zu - r=%zu - c=%zu - value=%.4f\n", pi, r, c, value);
        output[pi] = value;
    }
    
    free(irs);
    free(ics);
    return length;
}

size_t dtw_distances_matrix_parallel(dtwvalue *matrix, int nb_rows, int nb_cols, dtwvalue* output, DTWBlock* block, DTWSettings* settings) {
    // Requires openmp which is not supported for clang on mac by default (use newer version of clang)
    size_t r, c;
    size_t length;
    ssize_t *irs, *ics;

    if (dtw_distances_prepare(block, nb_rows, &irs, &ics, &length) != 0) {
        return 0;
    }

    size_t pi=0;
    #pragma omp parallel for private(pi, r, c)
    for(pi=0; pi<length; ++pi) {
//        printf("pi=%zu\n", pi);
        r = irs[pi];
        c = ics[pi];
        double value = dtw_distance(&matrix[r*nb_cols], nb_cols,
                                    &matrix[c*nb_cols], nb_cols, settings);
//        printf("pi=%zu - r=%zu->%zu - c=%zu - value=%.4f\n", pi, r, r*nb_cols, c, value);
        output[pi] = value;
    }
    
    free(irs);
    free(ics);
    return length;
}
