//
//  dd_loco.h
//  DTAIDistanceC
//
//  Created by Wannes Meert on 01/10/2024.
//  Copyright © 2024 Wannes Meert. All rights reserved.
//

#ifndef dd_loco_h
#define dd_loco_h

#include "dd_globals.h"
#include "dd_dtw.h"

/**
 Settings for local concurrences.
 
 @field `window`: Window size, expressed as distance to a single side including the diagonal, thus
 the total window size will be `window*2 - 1` and Euclidean distance is window=1.
 @field `only_triu`: Only the upper triangular values
 @field `gamma`: The gamma parameter in the formula
 @field `tau` :The tau parameter in the formula
 @field `delta` :The delta parameter in the formula
 @field `delta_factor`: The delta_factor parameter in the formula
 */
struct LoCoSettings_s {
    idx_t window;
    seq_t penalty;
    idx_t psi_1b;  // series 1, begin psi
    idx_t psi_2b;  // series 2, begin psi
    bool only_triu;
    seq_t gamma;
    seq_t tau;
    seq_t delta;
    seq_t delta_factor;
    StepType step_type;
};
typedef struct LoCoSettings_s LoCoSettings;


struct HeapEntry_s {
    idx_t idx;  // Index in wps array (including inf_rows and inf_cols
    seq_t val;
};
typedef struct HeapEntry_s HeapEntry;


LoCoSettings loco_settings_default(void);

seq_t loco_warping_paths(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, LoCoSettings *settings);
seq_t loco_warping_paths_typeI(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, LoCoSettings *settings);
seq_t loco_warping_paths_ndim_typeI(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim, LoCoSettings *settings);
seq_t loco_warping_paths_typeIII(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, LoCoSettings *settings);
seq_t loco_warping_paths_ndim_typeIII(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim, LoCoSettings *settings);

typedef struct {
  idx_t *array;
  size_t used;
  size_t size;
} BestPath;

void best_path_init(BestPath *a, size_t initialSize);
void best_path_insert(BestPath *a, idx_t element);
void best_path_free(BestPath *a);
BestPath loco_best_path(seq_t *wps, idx_t l1, idx_t l2, idx_t r, idx_t c, int min_size, LoCoSettings *settings);
BestPath loco_best_path_typeI(seq_t *wps, idx_t l1, idx_t l2, idx_t r, idx_t c, int min_size, LoCoSettings *settings);
BestPath loco_best_path_typeIII(seq_t *wps, idx_t l1, idx_t l2, idx_t r, idx_t c, int min_size, LoCoSettings *settings);


void loco_wps_argmax(seq_t *wps, idx_t l, idx_t *idxs, int n);

void loco_print_wps_type(seq_t * wps, idx_t l1, idx_t l2, LoCoSettings* settings);


#endif /* dd_loco_h */
