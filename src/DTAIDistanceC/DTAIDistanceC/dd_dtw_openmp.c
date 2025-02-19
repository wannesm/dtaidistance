/*!
@file dtw_openmp.c
@brief DTAIDistance.dtw

@author Wannes Meert
@copyright Copyright © 2020 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/

#include "dd_dtw_openmp.h"


bool is_openmp_supported() {
#if defined(_OPENMP)
    return true;
#else
    return false;
#endif
}


/**
 Check the arguments passed to dtw_distances_* and prepare the array of indices to be used.
 The indices are created upfront to allow for easy parallelization.
 
 @param block Block to indicate which series to compare.
 @param nb_series_r Number of series in the first matrix
 @param nb_series_c Number of series in the second matrix
 @param cbs Column begin indices for a row series index
 @param rls Location start for row in distances array
 @param length Length of (compact) distances matrix
 @param settings : Settings for DTW
 
 @return 0 if all is ok, other number if not.
 */
int dtw_distances_prepare(DTWBlock *block, idx_t nb_series_r, idx_t nb_series_c, idx_t **cbs, idx_t **rls, idx_t *length, DTWSettings *settings) {
    idx_t cb, rs, ir;
    
    *length = dtw_distances_length(block, nb_series_r, nb_series_c);
    if (length == 0) {
        return 1;
    }
    
    // Correct block
    if (block->re == 0) {
        block->re = nb_series_r;
    }
    if (block->ce == 0) {
        block->ce = nb_series_c;
    }
    if (block->re <= block->rb) {
        *length = 0;
        return 1;
    }
    if (block->ce <= block->cb) {
        *length = 0;
        return 1;
    }

    if (block->triu) {
        *cbs = (idx_t *)malloc(sizeof(idx_t) * (block->re - block->rb));
        if (!cbs) {
            printf("Error: dtw_distances_* - cannot allocate memory (cbs length = %zu)", block->re - block->rb);
            *length = 0;
            return 1;
        }
        *rls = (idx_t *)malloc(sizeof(idx_t) * (block->re - block->rb));
        if (!rls) {
            printf("Error: dtw_distances_* - cannot allocate memory (rls length = %zu)", block->re - block->rb);
            *length = 0;
            return 1;
        }
        ir = 0;
        rs = 0;
        assert(block->rb < block->re);
        for (idx_t r=block->rb; r<block->re; r++) {
            if (r + 1 > block->cb) {
                cb = r+1;
            } else {
                cb = block->cb;
            }
            (*cbs)[ir] = cb;
            (*rls)[ir] = rs;
            rs += block->ce - cb;
            ir += 1;
        }
    } else { // triu=false
        *cbs = NULL;
        *rls = NULL;
    }
    return 0;
}


/*!
Distance matrix for n-dimensional DTW, executed on a list of pointers to arrays and in parallel.

@see dtw_distances_ptrs
*/
idx_t dtw_distances_ptrs_parallel(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths,
                          seq_t* output, DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, r_i, c_i;
    idx_t length;
    idx_t *cbs, *rls;

    if (dtw_distances_prepare(block, nb_ptrs, nb_ptrs, &cbs, &rls, &length, settings) != 0) {
        return 0;
    }
    
#if defined(_OPENMP)
    r_i=0;
    // Rows have different lengths, thus use guided scheduling to make threads with shorter rows
    // not wait for threads with longer rows. Also the first rows are always longer than the last
    // ones (upper triangular matrix), so this nicely aligns with the guided strategy.
    // Using schedule("static, 1") is also fast for the same reason (neighbor rows are almost
    // the same length, thus a circular assignment works well) but assumes all DTW computations take
    // the same amount of time.
    #pragma omp parallel for private(r_i, c_i, r, c) schedule(guided)
    for (r_i=0; r_i < (block->re - block->rb); r_i++) {
        r = block->rb + r_i;
        c_i = 0;
        if (block->triu) {
            c = cbs[r_i];
        } else {
            c = block->cb;
        }
        for (; c<block->ce; c++) {
            double value = dtw_distance(ptrs[r], lengths[r],
                                        ptrs[c], lengths[c], settings);
            if (block->triu) {
                output[rls[r_i] + c_i] = value;
            } else {
                output[(block->ce - block->cb) * r_i + c_i] = value;
            }
            c_i++;
        }
    }
    
    if (block->triu) {
        free(cbs);
        free(rls);
    }
    return length;
#else
    printf("ERROR: DTAIDistanceC is compiled without OpenMP support.\n");
    for  (r_i=0; r_i<length; r_i++) {
        output[r_i] = 0;
    }
    return 0;
#endif
}


/*!
Distance matrix for n-dimensional DTW, executed on a list of pointers to arrays and in parallel.

@see dtw_distances_ndim_ptrs
*/
idx_t dtw_distances_ndim_ptrs_parallel(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths, int ndim,
                          seq_t* output, DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, r_i, c_i;
    idx_t length;
    idx_t *cbs, *rls;

    if (dtw_distances_prepare(block, nb_ptrs, nb_ptrs, &cbs, &rls, &length, settings) != 0) {
        return 0;
    }
    
#if defined(_OPENMP)
    r_i=0;
    #pragma omp parallel for private(r_i, c_i, r, c) schedule(guided)
    for (r_i=0; r_i < (block->re - block->rb); r_i++) {
        r = block->rb + r_i;
        c_i = 0;
        if (block->triu) {
            c = cbs[r_i];
        } else {
            c = block->cb;
        }
        for (; c<block->ce; c++) {
            double value = dtw_distance_ndim(ptrs[r], lengths[r],
                          ptrs[c], lengths[c],
                          ndim, settings);
            if (block->triu) {
                output[rls[r_i] + c_i] = value;
            } else {
                output[(block->ce - block->cb) * r_i + c_i] = value;
            }
            c_i++;
        }
    }
    
    if (block->triu) {
        free(cbs);
        free(rls);
    }
    return length;
#else
    printf("ERROR: DTAIDistanceC is compiled without OpenMP support.\n");
    for  (r_i=0; r_i<length; r_i++) {
        output[r_i] = 0;
    }
    return 0;
#endif
}


/*!
Distance matrix for n-dimensional DTW, executed on a 2-dimensional array and in parallel.

@see dtw_distances_matrix
*/
idx_t dtw_distances_matrix_parallel(seq_t *matrix, idx_t nb_rows, idx_t nb_cols,
                          seq_t* output, DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, r_i, c_i;
    idx_t length;
    idx_t *cbs, *rls;

    if (dtw_distances_prepare(block, nb_rows, nb_rows, &cbs, &rls, &length, settings) != 0) {
        return 0;
    }
    
#if defined(_OPENMP)
    r_i=0;
    #pragma omp parallel for private(r_i, c_i, r, c) schedule(guided)
    for (r_i=0; r_i < (block->re - block->rb); r_i++) {
        r = block->rb + r_i;
        c_i = 0;
        if (block->triu) {
            c = cbs[r_i];
        } else {
            c = block->cb;
        }
        for (; c<block->ce; c++) {
            double value = dtw_distance(&matrix[r*nb_cols], nb_cols,
                                         &matrix[c*nb_cols], nb_cols, settings);
            if (block->triu) {
                output[rls[r_i] + c_i] = value;
            } else {
                output[(block->ce - block->cb) * r_i + c_i] = value;
            }
            c_i++;
        }
    }
    
    if (block->triu) {
        free(cbs);
        free(rls);
    }
    return length;
#else
    printf("ERROR: DTAIDistanceC is compiled without OpenMP support.\n");
    for  (r_i=0; r_i<length; r_i++) {
        output[r_i] = 0;
    }
    return 0;
#endif
}


/*!
Distance matrix for n-dimensional DTW, executed on a 3-dimensional array and in parallel.

@see dtw_distances_ndim_matrix
*/
idx_t dtw_distances_ndim_matrix_parallel(seq_t *matrix, idx_t nb_rows, idx_t nb_cols, int ndim,
                          seq_t* output, DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, r_i, c_i;
    idx_t length;
    idx_t *cbs, *rls;

    if (dtw_distances_prepare(block, nb_rows, nb_rows, &cbs, &rls, &length, settings) != 0) {
        return 0;
    }
    
#if defined(_OPENMP)
    r_i=0;
    #pragma omp parallel for private(r_i, c_i, r, c) schedule(guided)
    for (r_i=0; r_i < (block->re - block->rb); r_i++) {
        r = block->rb + r_i;
        c_i = 0;
        if (block->triu) {
            c = cbs[r_i];
        } else {
            c = block->cb;
        }
        for (; c<block->ce; c++) {
            double value = dtw_distance_ndim(&matrix[r*nb_cols*ndim], nb_cols,
                                             &matrix[c*nb_cols*ndim], nb_cols,
                                             ndim, settings);
            if (block->triu) {
                output[rls[r_i] + c_i] = value;
            } else {
                output[(block->ce - block->cb) * r_i + c_i] = value;
            }
            c_i++;
        }
    }
    
    if (block->triu) {
        free(cbs);
        free(rls);
    }
    return length;
#else
    printf("ERROR: DTAIDistanceC is compiled without OpenMP support.\n");
    for  (r_i=0; r_i<length; r_i++) {
        output[r_i] = 0;
    }
    return 0;
#endif
}


/*!

@see dtw_distances_matrices
*/
idx_t dtw_distances_matrices_parallel(seq_t *matrix_r, idx_t nb_rows_r, idx_t nb_cols_r,
                          seq_t *matrix_c, idx_t nb_rows_c, idx_t nb_cols_c,
                          seq_t* output, DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, r_i, c_i;
    idx_t length;
    idx_t *cbs, *rls;

    if (dtw_distances_prepare(block, nb_rows_r, nb_rows_c, &cbs, &rls, &length, settings) != 0) {
        return 0;
    }
    
#if defined(_OPENMP)
    r_i=0;
    #pragma omp parallel for private(r_i, c_i, r, c) schedule(guided)
    for (r_i=0; r_i < (block->re - block->rb); r_i++) {
        r = block->rb + r_i;
        c_i = 0;
        if (block->triu) {
            c = cbs[r_i];
        } else {
            c = block->cb;
        }
        for (; c<block->ce; c++) {
            double value = dtw_distance(&matrix_r[r*nb_cols_r], nb_cols_r,
                                        &matrix_c[c*nb_cols_c], nb_cols_c, settings);
            if (block->triu) {
                output[rls[r_i] + c_i] = value;
            } else {
                output[(block->ce - block->cb) * r_i + c_i] = value;
            }
            c_i++;
        }
    }
    
    if (block->triu) {
        free(cbs);
        free(rls);
    }
    return length;
#else
    printf("ERROR: DTAIDistanceC is compiled without OpenMP support.\n");
    for  (r_i=0; r_i<length; r_i++) {
        output[r_i] = 0;
    }
    return 0;
#endif
}


/*!

@see dtw_distances_ndim_matrices
*/
idx_t dtw_distances_ndim_matrices_parallel(seq_t *matrix_r, idx_t nb_rows_r, idx_t nb_cols_r,
                          seq_t *matrix_c, idx_t nb_rows_c, idx_t nb_cols_c, int ndim,
                          seq_t* output, DTWBlock* block, DTWSettings* settings) {
    idx_t r, c, r_i, c_i;
    idx_t length;
    idx_t *cbs, *rls;

    if (dtw_distances_prepare(block, nb_rows_r, nb_rows_c, &cbs, &rls, &length, settings) != 0) {
        return 0;
    }
    
#if defined(_OPENMP)
    r_i=0;
    #pragma omp parallel for private(r_i, c_i, r, c) schedule(guided)
    for (r_i=0; r_i < (block->re - block->rb); r_i++) {
        r = block->rb + r_i;
        c_i = 0;
        if (block->triu) {
            c = cbs[r_i];
        } else {
            c = block->cb;
        }
        for (; c<block->ce; c++) {
            double value = dtw_distance_ndim(&matrix_r[r*nb_cols_r*ndim], nb_cols_r,
                                             &matrix_c[c*nb_cols_c*ndim], nb_cols_c,
                                             ndim, settings);
            if (block->triu) {
                output[rls[r_i] + c_i] = value;
            } else {
                output[(block->ce - block->cb) * r_i + c_i] = value;
            }
            c_i++;
        }
    }
    
    if (block->triu) {
        free(cbs);
        free(rls);
    }
    return length;
#else
    printf("ERROR: DTAIDistanceC is compiled without OpenMP support.\n");
    for  (r_i=0; r_i<length; r_i++) {
        output[r_i] = 0;
    }
    return 0;
#endif
}

