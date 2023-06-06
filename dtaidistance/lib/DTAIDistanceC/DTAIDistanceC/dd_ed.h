/*!
@header ed.h
@brief DTAIDistance.ed : Euclidean Distance

@author Wannes Meert
@copyright Copyright © 2020 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/

#ifndef ed_h
#define ed_h

#include <stdio.h>
#include <math.h>

#include "dd_globals.h"


#define SEDIST(x, y) ((x - y) * (x - y))


seq_t euclidean_distance(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2);
seq_t euclidean_distance_euclidean(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2);
seq_t euclidean_distance_ndim(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim);
seq_t euclidean_distance_ndim_euclidean(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim);

#endif /* ed_h */
