#include "dd_dtw.h"

void lower_upper_lemire(seq_t *t, int len, seq_t *l, seq_t *u, DTWSettings *settings);

seq_t lb_keogh_from_envelope(seq_t *s1, idx_t l1, seq_t *l, seq_t *u, DTWSettings *settings);

int ucrdtw(seq_t* data, long long data_size, int skip, seq_t* query, long query_size, int verbose, long long* location, double* distance, DTWSettings *settings);
