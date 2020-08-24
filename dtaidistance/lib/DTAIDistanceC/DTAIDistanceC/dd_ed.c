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

 @param s1 : Sequence of numbers
 @param s2 : Sequence of numbers
 @return Euclidean distance
 
 */
seq_t euclidean_distance(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2) {
    idx_t n = MIN(l1, l2);
    seq_t ub = 0;
    for (idx_t i=0; i<n; i++) {
        ub += EDIST(s1[i], s2[i]);
    }
    // If the two series differ in length, compare the last element of the shortest series
    // to the remaining elements in the longer series
    if (l1 > l2) {
        for (idx_t i=n; i<l1; i++) {
            ub += EDIST(s1[i], s2[n-1]);
        }
    } else if (l1 < l2) {
        for (idx_t i=n; i<l2; i++) {
            ub += EDIST(s1[n-1], s2[i]);
        }
    }
    return sqrt(ub);
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
    seq_t ub = 0;
    for (idx_t i=0; i<n; i++) {
        idx = i*ndim;
        for (int d=0; d<ndim; d++) {
            ub += EDIST(s1[idx + d], s2[idx + d]);
        }
    }
    // If the two series differ in length, compare the last element of the shortest series
    // to the remaining elements in the longer series
    if (l1 > l2) {
        for (idx_t i=n; i<l1; i++) {
            idx = i*ndim;
            for (int d=0; d<ndim; d++) {
                ub += EDIST(s1[idx + d], s2[(n-1)*ndim]);
            }
        }
    } else if (l1 < l2) {
        for (idx_t i=n; i<l2; i++) {
            idx = i*ndim;
            for (int d=0; d<ndim; d++) {
                ub += EDIST(s1[(n-1)*ndim], s2[idx + d]);
            }
        }
    }
    return sqrt(ub);
}

