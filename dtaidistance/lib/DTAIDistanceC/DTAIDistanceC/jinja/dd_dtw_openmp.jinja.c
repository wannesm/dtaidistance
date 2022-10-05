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


{% set suffix = 'ptrs' %}
{%- include 'dtw_distances_parallel.jinja.c' %}

{% set suffix = 'ndim_ptrs' %}
{%- include 'dtw_distances_parallel.jinja.c' %}

{% set suffix = 'matrix' %}
{%- include 'dtw_distances_parallel.jinja.c' %}

{% set suffix = 'ndim_matrix' %}
{%- include 'dtw_distances_parallel.jinja.c' %}

{% set suffix = 'matrices' %}
{%- include 'dtw_distances_parallel.jinja.c' %}

{% set suffix = 'ndim_matrices' %}
{%- include 'dtw_distances_parallel.jinja.c' %}

