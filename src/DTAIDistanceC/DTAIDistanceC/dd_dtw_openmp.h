/*!
@header dtw_openmp.h
@brief DTAIDistance.dtw

@author Wannes Meert
@copyright Copyright © 2020 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/

#ifndef dtw_openmp_h
#define dtw_openmp_h

#include <stdio.h>
#include <assert.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "dd_dtw.h"

bool is_openmp_supported(void);
int    dtw_distances_prepare(DTWBlock *block, idx_t nb_series_r, idx_t nb_series_c, 
                             idx_t **cbs, idx_t **rls, idx_t *length, DTWSettings *settings);
idx_t dtw_distances_ptrs_parallel(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths,
                                   seq_t* output, DTWBlock* block, DTWSettings* settings);
idx_t dtw_distances_ndim_ptrs_parallel(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths, int ndim, seq_t* output,
                                        DTWBlock* block, DTWSettings* settings);
idx_t dtw_distances_matrix_parallel(seq_t *matrix, idx_t nb_rows, idx_t nb_cols,
                                     seq_t* output, DTWBlock* block, DTWSettings* settings);
idx_t dtw_distances_ndim_matrix_parallel(seq_t *matrix, idx_t nb_rows, idx_t nb_cols, int ndim, seq_t* output,
                                          DTWBlock* block, DTWSettings* settings);
idx_t dtw_distances_matrices_parallel(seq_t *matrix_r, idx_t nb_rows_r, idx_t nb_cols_r,
                          seq_t *matrix_c, idx_t nb_rows_c, idx_t nb_cols_c,
                                      seq_t* output, DTWBlock* block, DTWSettings* settings);
idx_t dtw_distances_ndim_matrices_parallel(seq_t *matrix_r, idx_t nb_rows_r, idx_t nb_cols_r,
                          seq_t *matrix_c, idx_t nb_rows_c, idx_t nb_cols_c, int ndim,
                                           seq_t* output, DTWBlock* block, DTWSettings* settings);


#endif /* dtw_openmp_h */
