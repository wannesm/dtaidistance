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

#include "dtw.h"

int    dtw_distances_prepare(DTWBlock *block, size_t len_cur,
                             size_t **irs, size_t **ics, size_t *length, DTWSettings *settings);
size_t dtw_distances_ptrs_parallel(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths,
                                   dtwvalue* output, DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_ptrs_ndim_parallel(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths, int ndim, dtwvalue* output,
                                        DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_matrix_parallel(dtwvalue *matrix, size_t nb_rows, size_t nb_cols,
                                     dtwvalue* output, DTWBlock* block, DTWSettings* settings);

#endif /* dtw_openmp_h */
