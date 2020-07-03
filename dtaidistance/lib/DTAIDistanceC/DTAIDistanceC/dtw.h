/*!
@header dtw.h
@brief DTAIDistance.dtw

@author Wannes Meert
@copyright Copyright © 2020 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/

#ifndef dtw_h
#define dtw_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>


#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/**
 @var keepRunning
 @abstract Indicator to track if an interrupt occured requiring the algorithm to exit.
 */
static volatile int keepRunning = 1;
/**
 @var printPrecision
 @abstract Number of decimals to print when printing (partial) distances.
 */
static int printPrecision = 3;

/* The dtwvalue type can be customized by changing the typedef. */
typedef double dtwvalue;

/**
Settings for DTW operations:
 
@field window: Window size; expressed as distance to a single side, thus
       the total window size will be window*2 + 1.
@field max_dist: Maximal distance, exit early if no lower value is possible
       and return INFINITY.
@field max_step: Maximal stepsize, replace value with INFINITY if step is
       larger than max_step.
@field max_length_diff: Maximal difference in length between two series.
       If longer, return INFINITY.
@field penalty: Customize the cost for expansion or compression.
@field psi: Psi relaxation allows to ignore this number of entries at the beginning
       and/or end of both sequences.
@field use_ssize_t: Internal variable. Check if array size would exceed SIZE_MAX (true)
       or PTRDIFF_MAX (false).
       This can be used to be compatible with Python's Py_ssize_t when wrapping this
       library: https://www.python.org/dev/peps/pep-0353/
 */
struct DTWSettings_s {
    size_t window;
    dtwvalue max_dist;
    dtwvalue max_step;
    size_t max_length_diff;
    dtwvalue penalty;
    size_t psi;
    bool use_ssize_t;
};
typedef struct DTWSettings_s DTWSettings;

/**
 Block to restrict comparisons between series.
 
 @field rb Row begin
 @field re Row end
 @field cb Column begin
 @field ce Column end
 */
struct DTWBlock_s {
    size_t rb;
    size_t re;
    size_t cb;
    size_t ce;
};
typedef struct DTWBlock_s DTWBlock;


// Settings
DTWSettings dtw_settings_default(void);
void        dtw_settings_print(DTWSettings *settings);

// DTW
dtwvalue dtw_distance(dtwvalue *s1, size_t l1, dtwvalue *s2, size_t l2, DTWSettings *settings);
dtwvalue dtw_warping_paths(dtwvalue *wps, dtwvalue *s1, size_t l1, dtwvalue *s2, size_t l2, bool return_dtw, bool do_sqrt, DTWSettings *settings);

// Block
DTWBlock dtw_block_empty(void);
void     dtw_block_print(DTWBlock *block);
bool     dtw_block_is_valid(DTWBlock *block, size_t nb_series);

// Distance matrix
size_t dtw_distances_ptrs(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths, dtwvalue* output,
                          DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_matrix(dtwvalue *matrix, size_t nb_rows, size_t nb_cols, dtwvalue* output,
                            DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_length(DTWBlock *block, size_t nb_series, bool use_ssize_t);

// Auxiliary functions
void dtw_int_handler(int dummy);

void dtw_printprecision_set(int precision);
void dtw_printprecision_reset(void);

void dtw_print_wps(dtwvalue * wps, size_t l1, size_t l2);
void dtw_print_twoline(dtwvalue * dtw, size_t r, size_t c, size_t length, int i0, int i1, size_t skip, size_t skipp, size_t maxj, size_t minj);

#endif /* dtw_h */
