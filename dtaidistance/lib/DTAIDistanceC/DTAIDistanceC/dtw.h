//
//  dtw.h
//  DTAIDistance
//
//  Copyright © 2020 Wannes Meert.
//  Apache License, Version 2.0, see LICENSE for details.
//

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

static volatile int keepRunning = 1;
static int printPrecision = 3;

/* The dtwvalue type can be customized by changing the typedef. */
typedef double dtwvalue;

/* Settings for DTW operations:
 
 @param window: Window size; expressed as distance to a single side, thus
        the total window size will be window*2 + 1.
 @param max_dist: Maximal distance, exit early if no lower value is possible
        and return INFINITY.
 @param max_step: Maximal stepsize, replace value with INFINITY if step is
        larger than max_step.
 @param max_length_diff:
 @param penalty: Customize the cost for expansion or compression.
 @param psi: Psi relaxation allows to ignore this number of entries at the beginning
        and/or end of both sequences.
 @param use_ssize_t: Internal variable. Check if array size would exceed SIZE_MAX (true)
        or PTRDIFF_MAX (false).
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

struct DTWBlock_s {
    size_t rb;
    size_t re;
    size_t cb;
    size_t ce;
};
typedef struct DTWBlock_s DTWBlock;


// Settings
DTWSettings dtw_default_settings(void);
void dtw_print_settings(DTWSettings *settings);

// DTW
dtwvalue dtw_distance(dtwvalue *s1, size_t l1, dtwvalue *s2, size_t l2, DTWSettings *settings);
dtwvalue dtw_warping_paths(dtwvalue *wps, dtwvalue *s1, size_t l1, dtwvalue *s2, size_t l2, bool return_dtw, bool do_sqrt, DTWSettings *settings);

// Distance matrix
DTWBlock dtw_empty_block(void);
void dtw_print_block(DTWBlock *block);
bool dtw_block_is_valid(DTWBlock *block, size_t nb_series);
size_t dtw_distances_ptrs(dtwvalue **ptrs, size_t nb_ptrs, size_t* lengths, dtwvalue* output,
                          DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_matrix(dtwvalue *matrix, size_t nb_rows, size_t nb_cols, dtwvalue* output,
                            DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_length(DTWBlock *block, size_t nb_series, bool use_ssize_t);

// Auxiliary functions
void dtw_set_printprecision(int precision);
void dtw_reset_printprecision(void);

void dtw_print_wps(dtwvalue * wps, size_t l1, size_t l2);
void dtw_print_twoline(dtwvalue * dtw, size_t r, size_t c, size_t length, int i0, int i1, size_t skip, size_t skipp, size_t maxj, size_t minj);

#endif /* dtw_h */
