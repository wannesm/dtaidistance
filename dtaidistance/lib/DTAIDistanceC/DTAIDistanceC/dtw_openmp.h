//
//  dtw_openmp.h
//  DTAIDistanceC
//
//  Copyright © 2020 Wannes Meert.
//  Apache License, Version 2.0, see LICENSE for details.
//

#ifndef dtw_openmp_h
#define dtw_openmp_h

#include <stdio.h>
#include <omp.h>

#include "dtw.h"

int dtw_distances_prepare(DTWBlock *block, int len_cur,
                          ssize_t **irs, ssize_t **ics, size_t *length);
size_t dtw_distances_ptrs_parallel(dtwvalue **ptrs, int nb_ptrs, int* lengths,
                                   dtwvalue* output, DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_matrix_parallel(dtwvalue *matrix, int nb_rows, int nb_cols,
                                     dtwvalue* output, DTWBlock* block, DTWSettings* settings);

#endif /* dtw_openmp_h */
