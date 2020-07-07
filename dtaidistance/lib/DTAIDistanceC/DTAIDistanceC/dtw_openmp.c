/*!
@file dtw_openmp.c
@brief DTAIDistance.dtw

@author Wannes Meert
@copyright Copyright © 2020 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/

#include "dtw_openmp.h"

/**
 Check the arguments passed to dtw_distances_* and prepare the array of indices to be used.
 The indices are created upfront to allow for easy parallelization.
 
 @param block Block to indicate which series to compare.
 @param nb_series Number of series
 @param irs Row indices
 @param ics  Column indices
 @param length Length of (compact) distances matrix
 @param settings : Settings for DTW
 
 @return 0 if all is ok, other number if not.
 */
int dtw_distances_prepare(DTWBlock *block, size_t nb_series, size_t **irs, size_t **ics, size_t *length, DTWSettings *settings) {
    size_t cb, ir;
    
    *length = dtw_distances_length(block, nb_series, settings->use_ssize_t);
    if (length == 0) {
        return 1;
    }
    
    // Correct block
    if (block->re == 0) {
        block->re = nb_series;
    }
    if (block->ce == 0) {
        block->ce = nb_series;
    }
    if (block->re <= block->rb) {
        *length = 0;
        return 1;
    }
    if (block->ce <= block->cb) {
        *length = 0;
        return 1;
    }

    *irs = (size_t *)malloc(sizeof(size_t) * *length);
    if (!irs) {
        printf("Error: dtw_distances_* - cannot allocate memory (length = %zu)", *length);
        *length = 0;
        return 1;
    }
    *ics = (size_t *)malloc(sizeof(size_t) * *length);
    if (!ics) {
        free(*irs);
        printf("Error: dtw_distances_* - cannot allocate memory (length = %zu)", *length);
        *length = 0;
        return 1;
    }
    ir = 0;
    assert(block->rb < block->re);
    for (size_t r=block->rb; r<block->re; r++) {
        if (r + 1 > block->cb) {
            cb = r+1;
        } else {
            cb = block->cb;
        }
        for (size_t c=cb; c<block->ce; c++) {
            (*irs)[ir] = r;
            (*ics)[ir] = c;
            ir += 1;
        }
    }
    assert(ir == *length);
    return 0;
}


size_t dtw_distances_ptrs_parallel(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths, dtwvalue* output,
                     DTWBlock* block, DTWSettings* settings) {
    // Requires openmp which is not supported for clang on mac by default (use newer version of clang)
    size_t r, c;
    size_t length;
    size_t *irs, *ics;

    if (dtw_distances_prepare(block, nb_ptrs, &irs, &ics, &length, settings) != 0) {
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


size_t dtw_distances_ptrs_ndim_parallel(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths, int ndim, dtwvalue* output,
                                        DTWBlock* block, DTWSettings* settings) {
    // Requires openmp which is not supported for clang on mac by default (use newer version of clang)
    size_t r, c;
    size_t length;
    size_t *irs, *ics;

    if (dtw_distances_prepare(block, nb_ptrs, &irs, &ics, &length, settings) != 0) {
        return 0;
    }

    size_t pi=0;
    #pragma omp parallel for private(pi, r, c)
    for(pi=0; pi<length; ++pi) {
//        printf("pi=%zu\n", pi);
        r = irs[pi];
        c = ics[pi];
        double value = dtw_distance_ndim(ptrs[r], lengths[r],
                                         ptrs[c], lengths[c],
                                         ndim, settings);
//        printf("pi=%zu - r=%zu - c=%zu - value=%.4f\n", pi, r, c, value);
        output[pi] = value;
    }
    
    free(irs);
    free(ics);
    return length;
}

size_t dtw_distances_matrix_parallel(dtwvalue *matrix, size_t nb_rows, size_t nb_cols, dtwvalue* output, DTWBlock* block, DTWSettings* settings) {
    // Requires openmp which is not supported for clang on mac by default (use newer version of clang)
    size_t r, c;
    size_t length;
    size_t *irs, *ics;

    if (dtw_distances_prepare(block, nb_rows, &irs, &ics, &length, settings) != 0) {
        return 0;
    }

    size_t pi=0;
    assert(length > 0);
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
