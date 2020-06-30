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
 
 @remark We use int instead of size_t because we need operations on the indices that return
         negative results (e.g. negative ranges).
 */
struct DTWSettings_s {
    int window;
    dtwvalue max_dist;
    dtwvalue max_step;
    int max_length_diff;
    dtwvalue penalty;
    int psi;
};
typedef struct DTWSettings_s DTWSettings;

struct DTWBlock_s {
    int rb;
    int re;
    int cb;
    int ce;
};
typedef struct DTWBlock_s DTWBlock;


// DTW
DTWSettings dtw_default_settings(void);
dtwvalue dtw_distance(dtwvalue *s1, int l1, dtwvalue *s2, int l2, DTWSettings *settings);
dtwvalue dtw_warping_paths(dtwvalue *wps, dtwvalue *s1, int l1, dtwvalue *s2, int l2, bool return_dtw, bool do_sqrt, DTWSettings *settings);

// Distance matrix
DTWBlock dtw_empty_block(void);
size_t dtw_distances_ptrs(dtwvalue **ptrs, int nb_ptrs, int* lengths, dtwvalue* output,
                          DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_matrix(dtwvalue *matrix, int nb_rows, int nb_cols, dtwvalue* output,
                            DTWBlock* block, DTWSettings* settings);
size_t dtw_distances_length(DTWBlock *block, int nb_series);

// Auxiliary functions
void dtw_set_printprecision(int precision);
void dtw_reset_printprecision(void);

void dtw_print_wps(dtwvalue * wps, int l1, int l2);
void dtw_print_twoline(dtwvalue * dtw, int r, int c, int length, int i0, int i1, int skip, int skipp, int maxj, int minj);
void dtw_print_settings(DTWSettings *settings);

#endif /* dtw_h */
