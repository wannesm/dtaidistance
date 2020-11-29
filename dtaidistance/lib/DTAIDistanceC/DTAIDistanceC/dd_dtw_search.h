#include "dd_dtw.h"


void lower_upper_naive(seq_t *t, int len, seq_t *l, seq_t *u, DTWSettings *settings);
void lower_upper_lemire(seq_t *t, int len, seq_t *l, seq_t *u, DTWSettings *settings);
void lower_upper_ascending_minima(seq_t *t, int len, seq_t *l, seq_t *u, DTWSettings *settings);

seq_t lb_keogh_from_envelope(seq_t *s1, idx_t l1, seq_t *l, seq_t *u, DTWSettings *settings);

int nn_lb_keogh(seq_t *data, idx_t data_size, int skip, seq_t *query, seq_t *L, seq_t *U, idx_t query_size, int verbose, idx_t *location, seq_t *distance, DTWSettings *settings);

int ucrdtw(seq_t* data, long long data_size, int skip, seq_t* query, long query_size, int verbose, long long* location, double* distance, DTWSettings *settings);
