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
#include <omp.h>

#include "dd_dtw.h"

int    dtw_distances_prepare(DTWBlock *block, size_t nb_series,
                             size_t **cbs, size_t **rls, size_t *length, DTWSettings *settings);
size_t dtw_distances_ptrs_parallel(seq_t **ptrs, size_t nb_ptrs, size_t* lengths,
                                   seq_t* output, DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_ndim_ptrs_parallel(seq_t **ptrs, size_t nb_ptrs, size_t* lengths, int ndim, seq_t* output,
                                        DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_matrix_parallel(seq_t *matrix, size_t nb_rows, size_t nb_cols,
                                     seq_t* output, DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_ndim_matrix_parallel(seq_t *matrix, size_t nb_rows, size_t nb_cols, int ndim, seq_t* output,
                                          DTWBlock* block, DTWSettings* settings);


#endif /* dtw_openmp_h */
