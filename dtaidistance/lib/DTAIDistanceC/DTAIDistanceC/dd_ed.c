//
//  ed.c
//  DTAIDistanceC
//
//  Created by Wannes Meert on 13/07/2020.
//  Copyright © 2020 Wannes Meert. All rights reserved.
//

#include "dd_ed.h"




/*!
Euclidean distance between two sequences of values, can differ in length.

If the two series differ in length, compare the last element of the shortest series
to the remaining elements in the longer series. This is compatible with Euclidean
distance being used as an upper bound for DTW.

@param s1 : Sequence of numbers.
@param s2 : Sequence of numbers.
@return Euclidean distance
*/
seq_t euclidean_distance(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2) {
    idx_t n = MIN(l1, l2);
    seq_t ub = 0;
    for (idx_t i=0; i<n; i++) {
        ub += SEDIST(s1[i], s2[i]);
    }
    // If the two series differ in length, compare the last element of the shortest series
    // to the remaining elements in the longer series
    if (l1 > l2) {
        for (idx_t i=n; i<l1; i++) {
            ub += SEDIST(s1[i], s2[n-1]);
        }
    } else if (l1 < l2) {
        for (idx_t i=n; i<l2; i++) {
            ub += SEDIST(s1[n-1], s2[i]);
        }
    }
    ub = sqrt(ub);
    return ub;
}



/*!
Euclidean distance between two sequences of values, can differ in length.

If the two series differ in length, compare the last element of the shortest series
to the remaining elements in the longer series. This is compatible with Euclidean
distance being used as an upper bound for DTW.

@param s1 : Sequence of numbers.
@param s2 : Sequence of numbers.
@return Euclidean distance
*/
seq_t euclidean_distance_euclidean(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2) {
    idx_t n = MIN(l1, l2);
    seq_t ub = 0;
    for (idx_t i=0; i<n; i++) {
        ub += fabs(s1[i] - s2[i]);
    }
    // If the two series differ in length, compare the last element of the shortest series
    // to the remaining elements in the longer series
    if (l1 > l2) {
        for (idx_t i=n; i<l1; i++) {
            ub += fabs(s1[i] - s2[n-1]);
        }
    } else if (l1 < l2) {
        for (idx_t i=n; i<l2; i++) {
            ub += fabs(s1[n-1] - s2[i]);
        }
    }
    return ub;
}



/*!
Euclidean distance between two sequences of values, can differ in length.

If the two series differ in length, compare the last element of the shortest series
to the remaining elements in the longer series. This is compatible with Euclidean
distance being used as an upper bound for DTW.
The sequences represent a sequence of n-dimensional vectors. The array is
assumed to be c-contiguous with as 1st dimension the sequence and the
2nd dimension the n-dimensional vector.

@param s1 : Sequence of numbers.
@param s2 : Sequence of numbers.
@return Euclidean distance
*/
seq_t euclidean_distance_ndim(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim) {
    idx_t n = MIN(l1, l2);
    idx_t idx;
    seq_t d;
    seq_t ub = 0;
    for (idx_t i=0; i<n; i++) {
        idx = i*ndim;
        d = 0;
        for (int d=0; d<ndim; d++) {
            d += SEDIST(s1[idx + d], s2[idx + d]);
        }
        ub += d;
    }
    // If the two series differ in length, compare the last element of the shortest series
    // to the remaining elements in the longer series
    if (l1 > l2) {
        for (idx_t i=n; i<l1; i++) {
            idx = i*ndim;
            d = 0;
            for (int d=0; d<ndim; d++) {
                d += SEDIST(s1[idx + d], s2[(n-1)*ndim]);
            }
            ub += d;
        }
    } else if (l1 < l2) {
        for (idx_t i=n; i<l2; i++) {
            idx = i*ndim;
            d = 0;
            for (int d=0; d<ndim; d++) {
                d += SEDIST(s1[(n-1)*ndim], s2[idx + d]);
            }
            ub += d;
        }
    }
    ub = sqrt(ub);
    return ub;
}



/*!
Euclidean distance between two sequences of values, can differ in length.

If the two series differ in length, compare the last element of the shortest series
to the remaining elements in the longer series. This is compatible with Euclidean
distance being used as an upper bound for DTW.
The sequences represent a sequence of n-dimensional vectors. The array is
assumed to be c-contiguous with as 1st dimension the sequence and the
2nd dimension the n-dimensional vector.

@param s1 : Sequence of numbers.
@param s2 : Sequence of numbers.
@return Euclidean distance
*/
seq_t euclidean_distance_ndim_euclidean(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim) {
    idx_t n = MIN(l1, l2);
    idx_t idx;
    seq_t d;
    seq_t ub = 0;
    for (idx_t i=0; i<n; i++) {
        idx = i*ndim;
        d = 0;
        for (int d=0; d<ndim; d++) {
            d += SEDIST(s1[idx + d], s2[idx + d]);
        }
        d = sqrt(d);
        ub += d;
    }
    // If the two series differ in length, compare the last element of the shortest series
    // to the remaining elements in the longer series
    if (l1 > l2) {
        for (idx_t i=n; i<l1; i++) {
            idx = i*ndim;
            d = 0;
            for (int d=0; d<ndim; d++) {
                d += SEDIST(s1[idx + d], s2[(n-1)*ndim]);
            }
            d = sqrt(d);
            ub += d;
        }
    } else if (l1 < l2) {
        for (idx_t i=n; i<l2; i++) {
            idx = i*ndim;
            d = 0;
            for (int d=0; d<ndim; d++) {
                d += SEDIST(s1[(n-1)*ndim], s2[idx + d]);
            }
            d = sqrt(d);
            ub += d;
        }
    }
    return ub;
}
