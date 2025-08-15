/*!
@file dtw.c
@brief DTAIDistance.loco

@author Wannes Meert
@copyright Copyright © 2024 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/
#include "dd_loco.h"


//#define DTWDEBUG

// MARK: Settings

LoCoSettings loco_settings_default(void) {
    LoCoSettings s = {
        .window = 0,
        .penalty = 0,
        .psi_1b = 0,
        .psi_2b = 0,
        .only_triu = false,
        .gamma = 1,
        .tau = 0,
        .delta = 0,
        .delta_factor = 1,
        .step_type = TypeIII
    };
    return s;
}

// MARK: Warping paths

seq_t loco_warping_paths(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, LoCoSettings *settings) {
    const int ndim = 1;
    switch (settings->step_type) {
        case TypeI:
            return loco_warping_paths_ndim_typeI(wps, s1, l1, s2, l2, ndim, settings);
        case TypeIII:
            return loco_warping_paths_ndim_typeIII(wps, s1, l1, s2, l2, ndim, settings);
    }
    return 0;
}


seq_t loco_warping_paths_typeI(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2,
                                     LoCoSettings *settings) {
    return loco_warping_paths_ndim_typeI(
               wps, s1, l1, s2, l2, 1, settings);
}

/*!
 Compute the warping paths / cumulative cost matrix for Local Concurrences.
 Using the step function: (1, 0), (1, 1), (0, 1)

 @param wps Compact warping paths matrix
 @param s1 Series 1
 @param l1 Length of series 1
 @param s2 Series 2
 @param l2 Length of series 2
 @param ndim The dimensions of series 1 and series 2
 @param settings A DTWSettings object
 */
seq_t loco_warping_paths_ndim_typeI(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2,
                                          int ndim, LoCoSettings *settings) {
    // TODO: no support for window
    assert(settings->window == 0 || settings->window == MAX(l1, l2));
    // TODO: no support for only_triu
    assert(!settings->only_triu);
    const idx_t inf_cols = 1;
    const idx_t inf_rows = 1;
    idx_t width = l2 + inf_cols;
    const seq_t penalty = settings->penalty;
    const seq_t gamma = settings->gamma;
    const seq_t tau = settings->tau;
    const seq_t delta = settings->delta;
    const seq_t delta_factor = settings->delta_factor;

    idx_t ri, ci, wpsi;
    seq_t d;
    idx_t ri_idx, ci_idx;

    // First rows
    wpsi = 0;
    for (ri=0; ri<inf_rows; ri++) {
        for (ci=0; ci<inf_cols+settings->psi_2b; ci++) {
            wps[wpsi] = 0;
            wpsi++;
        }
        for (; ci<width; ci++) {
            wps[wpsi] = -INFINITY;
            wpsi++;
        }
    }
    // First columns
    wpsi = inf_rows*width;
    for (; ri<inf_rows+settings->psi_1b; ri++) {
        for (ci=0; ci<inf_cols; ci++) {
            wps[wpsi] = 0;
            wpsi++;
        }
        wpsi += width - inf_cols + 1;
    }
    for (; ri<inf_rows+l1; ri++) {
        for (ci=0; ci<inf_cols; ci++) {
            wps[wpsi] = -INFINITY;
            wpsi++;
        }
        wpsi += width - inf_cols;
    }

    // Cumulative costs
    idx_t ri_widthp = 0;
    idx_t ri_width = width;
    seq_t dtw_prev;
    for (ri=0; ri<l1; ri++) {
        ri_idx = ri * ndim;
        wpsi = inf_cols;
        for (ci=0; ci<l2; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += SEDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            d = exp(-gamma * d);
            // Steps: (0, 1), (1, 1), (1, 0)
            dtw_prev = MAX3(wps[ri_width  + wpsi - 1] - penalty,  // left
                            wps[ri_widthp + wpsi - 1],            // diagonal
                            wps[ri_widthp + wpsi]     - penalty); // up
            if (d < tau) {
                dtw_prev = delta + delta_factor * dtw_prev;
            } else {
                dtw_prev = d + dtw_prev;
            }
            if (dtw_prev < 0) {
                dtw_prev = 0;
            }
            wps[ri_width + wpsi] = dtw_prev;
            wpsi++;
        }
        ri_widthp = ri_width;
        ri_width += width;
    }

    // Nothing to return
    return 0;
}


seq_t loco_warping_paths_typeIII(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2,
                                     LoCoSettings *settings) {
    return loco_warping_paths_ndim_typeIII(
               wps, s1, l1, s2, l2, 1, settings);
}

/*!
 Compute the warping paths / cumulative cost matrix for Local Concurrences.
 Using the step function: (2, 1), (1, 1), (1, 2)

 @param wps Compact warping paths matrix
 @param s1 Series 1
 @param l1 Length of series 1
 @param s2 Series 2
 @param l2 Length of series 2
 @param ndim The dimensions of series 1 and series 2
 @param settings A DTWSettings object
 */
seq_t loco_warping_paths_ndim_typeIII(seq_t *wps, seq_t *s1, idx_t l1, seq_t *s2, idx_t l2,
                                          int ndim, LoCoSettings *settings) {
    // TODO: no support for window
    assert(settings->window == 0 || settings->window == MAX(l1, l2));
    // TODO: no support for only_triu
    assert(!settings->only_triu);
    const idx_t inf_cols = 2;
    const idx_t inf_rows = 2;
    idx_t width = l2 + inf_cols;
    const seq_t penalty = settings->penalty;
    const seq_t gamma = settings->gamma;
    const seq_t tau = settings->tau;
    const seq_t delta = settings->delta;
    const seq_t delta_factor = settings->delta_factor;

    idx_t ri, ci, wpsi;
    seq_t d;
    idx_t ri_idx, ci_idx;

    // First rows
    wpsi = 0;
    for (ri=0; ri<inf_rows; ri++) {
        for (ci=0; ci<inf_cols+settings->psi_2b; ci++) {
            wps[wpsi] = 0;
            wpsi++;
        }
        for (; ci<width; ci++) {
            wps[wpsi] = -INFINITY;
            wpsi++;
        }
    }
    // First columns
    wpsi = inf_rows*width;
    for (; ri<inf_rows+settings->psi_1b; ri++) {
        for (ci=0; ci<inf_cols; ci++) {
            wps[wpsi] = 0;
            wpsi++;
        }
        wpsi += width - inf_cols + 1;
    }
    for (; ri<inf_rows+l1; ri++) {
        for (ci=0; ci<inf_cols; ci++) {
            wps[wpsi] = -INFINITY;
            wpsi++;
        }
        wpsi += width - inf_cols;
    }

    // Cumulative costs
    idx_t ri_widthpp = 0;
    idx_t ri_widthp = width;
    idx_t ri_width = inf_rows*width;
    seq_t dtw_prev;
    for (ri=0; ri<l1; ri++) {
        ri_idx = ri * ndim;
        wpsi = inf_cols;
        for (ci=0; ci<l2; ci++) {
            ci_idx = ci * ndim;
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += SEDIST(s1[ri_idx + d_i], s2[ci_idx + d_i]);
            }
            d = exp(-gamma * d);
            // Steps: (1, 2), (1, 1), (2, 1)
            dtw_prev = MAX3(wps[ri_widthp  + wpsi - 2] - penalty,  // left
                            wps[ri_widthp  + wpsi - 1],            // diagonal
                            wps[ri_widthpp + wpsi - 1] - penalty); // up
            if (d < tau) {
                dtw_prev = delta + delta_factor * dtw_prev;
            } else {
                dtw_prev = d + dtw_prev;
            }
            if (dtw_prev < 0) {
                dtw_prev = 0;
            }
            wps[ri_width + wpsi] = dtw_prev;
            wpsi++;
        }
        ri_widthpp = ri_widthp;
        ri_widthp = ri_width;
        ri_width += width;
    }

    // Nothing to return
    return 0;
}


// MARK: Best path

void best_path_init(BestPath *a, size_t initialSize) {
    a->array = malloc(initialSize * sizeof(idx_t));
    if (a->array == NULL) {
        printf("\ERROR: cannot allocate memory for LoCo best path calculation");
        exit(1);
    }
    a->used = 0;
    a->size = initialSize;
}

void best_path_insert(BestPath *a, idx_t element) {
    idx_t* array_old;
    if (a->used == a->size) {
        a->size *= 2;
        array_old = a->array;
        a->array = realloc(a->array, a->size * sizeof(idx_t));
        if (a->array == NULL) {
            printf("\ERROR: cannot allocate memory for LoCo best path calculation");
            free(array_old);
            exit(1);
        }
    }
    a->array[a->used++] = element;
}

void best_path_free(BestPath *a) {
  free(a->array);
  a->array = NULL;
  a->used = a->size = 0;
}

BestPath loco_best_path(seq_t *wps, idx_t l1, idx_t l2, idx_t r, idx_t c, int init_size, LoCoSettings *settings) {
    switch (settings->step_type) {
        case TypeI:
            return loco_best_path_typeI(wps, l1, l2, r, c, init_size, settings);
            break;
        case TypeIII:
            return loco_best_path_typeIII(wps, l1, l2, r, c, init_size, settings);
            break;
    }
    return (BestPath){.array = NULL, .used = 0, .size = 0};
}


BestPath loco_best_path_typeI(seq_t *wps, idx_t l1, idx_t l2, idx_t r, idx_t c, int init_size, LoCoSettings *settings) {

    // Diagonal, left, up
    const size_t inf_rows = 1;
    const size_t inf_cols = 1;
    const size_t width = l2 + inf_cols;
    const size_t d_r[] = {1, 0, 1};
    const size_t d_c[] = {1, 1, 0};
    const size_t d_f[] = {width + 1, 1, width};

    BestPath bp;
    seq_t values[3];
    best_path_init(&bp, init_size);
    const seq_t penalty = settings->penalty;
    
    size_t i = r;
    size_t j = c;
    size_t f = r*width + c;
    while (i >= inf_rows && j >= inf_cols) {
        best_path_insert(&bp, f);
        values[0] = wps[f - d_f[0]]; // diagonal
        values[1] = wps[f - d_f[1]]; // left
        values[2] = wps[f - d_f[2]]; // up
        
        if (values[0] < 0) {
            values[0] = -1;
        }
        if (values[1] < 0) {
            values[1] = -1;
        } else {
            values[1] += penalty;
        }
        if (values[2] < 0) {
            values[2] = -1;
        } else {
            values[2] += penalty;
        }
        
        if (values[0] >= values[1] && values[0] >= values[2]) {
            if (values[0] <= 0) {
                break;
            }
            i -= d_r[0];
            j -= d_c[0];
            f -= d_f[0];
        } else if (values[1] > values[2]) {
            if (values[1] <= 0) {
                break;
            }
            i -= d_r[1];
            j -= d_c[1];
            f -= d_f[1];
        } else {
            if (values[2] <= 0) {
                break;
            }
            i -= d_r[2];
            j -= d_c[2];
            f -= d_f[2];
        }
    }
    if (i < inf_rows || j < inf_cols) {
        bp.used -= 1;
    }
    
    return bp;
}


BestPath loco_best_path_typeIII(seq_t *wps, idx_t l1, idx_t l2, idx_t r, idx_t c, int init_size, LoCoSettings *settings) {

    // Diagonal, left, up
    const size_t inf_rows = 2;
    const size_t inf_cols = 2;
    const size_t width = l2 + inf_cols;
    const size_t d_r[] = {1, 1, 2};
    const size_t d_c[] = {1, 2, 1};
    const size_t d_f[] = {width + 1, width + 2, 2*width + 1};

    BestPath bp;
    seq_t values[3];
    best_path_init(&bp, init_size);
    const seq_t penalty = settings->penalty;
    
    size_t i = r;
    size_t j = c;
    size_t f = r*width + c;
    while (i >= inf_rows && j >= inf_cols) {
        best_path_insert(&bp, f);
        values[0] = wps[f - d_f[0]]; // diagonal
        values[1] = wps[f - d_f[1]]; // left
        values[2] = wps[f - d_f[2]]; // up
        
        if (values[0] < 0) {
            values[0] = -1;
        }
        if (values[1] < 0) {
            values[1] = -1;
        } else {
            values[1] += penalty;
        }
        if (values[2] < 0) {
            values[2] = -1;
        } else {
            values[2] += penalty;
        }
        
        if (values[0] >= values[1] && values[0] >= values[2]) {
            if (values[0] <= 0) {
                break;
            }
            i -= d_r[0];
            j -= d_c[0];
            f -= d_f[0];
        } else if (values[1] > values[2]) {
            if (values[1] <= 0) {
                break;
            }
            i -= d_r[1];
            j -= d_c[1];
            f -= d_f[1];
        } else {
            if (values[2] <= 0) {
                break;
            }
            i -= d_r[2];
            j -= d_c[2];
            f -= d_f[2];
        }
    }
    if (i < inf_rows || j < inf_cols) {
        bp.used -= 1;
    }
    
    return bp;
}


// MARK: WPS Operations

int comp(const void * elem1, const void * elem2)
{
    HeapEntry *f = (HeapEntry*)elem1;
    HeapEntry *s = (HeapEntry*)elem2;
    if (f->val < s->val) return  1;
    if (f->val > s->val) return -1;
    return 0;
}

void loco_wps_argmax(seq_t *wps, idx_t l, idx_t *idxs, int n) {
    int i, j, k;
    HeapEntry swap;
    HeapEntry heap[n];
                             
    for (i=0; i<n; i++) {
        heap[i] = (HeapEntry){.idx=l+1, .val=0};
    }
    
    // Heap selection (Algorithms in C++, p348)
    // Min-heap: root is smallest entry in heap
    for (i=0; i<l; i++) {
        if (wps[i] > heap[0].val) {
            heap[0] = (HeapEntry){.idx=i, .val=wps[i]};
            for (j=0;;) {
                k = (j << 1) + 1;
                if (k > n-1) break;
                if (k != (n-1) && heap[k].val  > heap[k+1].val) k++;
                if (heap[j].val <= heap[k].val) break;
                swap = heap[k];
                heap[k] = heap[j];
                heap[j] = swap;
                j = k;
            }
        }
    }
    
    qsort(heap, n, sizeof(HeapEntry), comp);
    for (i=0; i<n; i++) {
        idxs[i] = heap[i].idx;
    }
}


void loco_print_wps_type(seq_t * wps, idx_t l1, idx_t l2, LoCoSettings* settings) {
    idx_t inf_cols = 1;
    idx_t inf_rows = 1;

    switch (settings->step_type) {
        case TypeI:
            break;
        case TypeIII:
            inf_cols = 2;
            inf_rows = 2;
            break;
    }
    
    idx_t width = l2 + inf_cols;
    idx_t height = l1 + inf_rows;
    idx_t ri, ci, wpsi;

    wpsi = 0;
    printf(" [[ ");
    for (ci=0; ci<inf_cols; ci++) {
        print_nb(wps[wpsi]);
        printf("_ ");
        wpsi++;
    }
    for (; ci<width; ci++) {
        print_nb(wps[wpsi]);
        printf("  ");
        wpsi++;
    }
    printf("]\n");
    for (ri=1; ri<height-1; ri++) {
        printf("  [ ");
        for (ci=0; ci<inf_cols; ci++) {
            print_nb(wps[wpsi]);
            printf("_ ");
            wpsi++;
        }
        for (; ci<width; ci++) {
            print_nb(wps[wpsi]);
            printf("  ");
            wpsi++;
        }
        printf("]\n");
    }
    printf("  [ ");
    for (ci=0; ci<inf_cols; ci++) {
        print_nb(wps[wpsi]);
        printf("_ ");
        wpsi++;
    }
    for (; ci<width; ci++) {
        print_nb(wps[wpsi]);
        printf("  ");
        wpsi++;
    }
    printf("]]\n");
}
