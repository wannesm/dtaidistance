/**
{%- if "ndim" in suffix %}
Compute the DTW between two n-dimensional series.
{%- else %}
Compute the DTW between two series.
{%- endif %}

{%- if "euclidean" == inner_dist %}
Use the Euclidean inner distance.
{%- else %}{# inner_dist == "squaredeuclidean"  #}
Use the Squared Euclidean inner distance.
{%- endif %}

@param s1 First sequence
@param l1 Length of first sequence. {% if "ndim" in suffix %}In tuples, real length should be length*ndim.{% endif %}
@param s2 Second sequence
@param l2 Length of second sequence. {% if "ndim" in suffix %}In tuples, real length should be length*ndim.{% endif %}
{%- if "ndim" in suffix %}
@param ndim Number of dimensions
{%- endif %}
@param settings A DTWSettings struct with options for the DTW algorithm.
*/
{%- if "ptrs" in suffix %}
{%- set i="i" %}
{%- set j="j" %}
{%- elif "matrix" in suffix %}
{%- set i="i_idx" %}
{%- set j="j_idx" %}
{%- endif %}
{%- if "euclidean" == inner_dist %}
{%- set suffix2="_euclidean" %}
{%- else %}
{%- set suffix2="" %}
{%- endif %}
seq_t dtw_distance{{ suffix }}{{ suffix2 }}(seq_t *s1, idx_t l1,
                      seq_t *s2, idx_t l2, {% if "ndim" in suffix %}int ndim,{% endif %}
                      DTWSettings *settings) {
    {%- if inner_dist != "euclidean" %}
    if (settings->inner_dist == 1) {
        return dtw_distance{{ suffix }}_euclidean(s1, l1, s2, l2, {% if "ndim" in suffix %}ndim, {% endif %} settings);
    }
    {%- endif %}
    assert(settings->psi_1b <= l1 && settings->psi_1e <= l1 &&
           settings->psi_2b <= l2 && settings->psi_2e <= l2);
    idx_t ldiff;
    idx_t dl;
    // DTWPruned
    idx_t sc = 0;
    idx_t ec = 0;
    bool smaller_found;
    idx_t ec_next;
    // signal(SIGINT, dtw_int_handler); // not compatible with OMP

    idx_t window = settings->window;
    seq_t max_step = settings->max_step;
    seq_t max_dist = settings->max_dist;
    seq_t penalty = settings->penalty;

    #ifdef DTWDEBUG
    printf("r=%zu, c=%zu\n", l1, l2);
    #endif
    if (settings->use_pruning || settings->only_ub) {
        {%- if "ndim" in suffix %}
        max_dist = ub_euclidean_ndim{{ suffix2 }}(s1, l1, s2, l2, ndim);
        {%- else %}
        max_dist = ub_euclidean{{ suffix2 }}(s1, l1, s2, l2);
        {%- endif %}
        {%- if "euclidean" == inner_dist %}
        {%- else %}
        max_dist = pow(max_dist, 2);
        {%- endif %}
        if (settings->only_ub) {
            return max_dist;
        }
    } else if (max_dist == 0) {
        max_dist = INFINITY;
    } else {
        max_dist = pow(max_dist, 2);
    }
    if (l1 > l2) {
        ldiff = l1 - l2;
        dl = ldiff;
    } else {
        ldiff  = l2 - l1;
        dl = 0;
    }
    if (settings->max_length_diff != 0 && ldiff > settings->max_length_diff) {
        return INFINITY;
    }
    if (window == 0) {
        window = MAX(l1, l2);
    }
    if (max_step == 0) {
        max_step = INFINITY;
    } else {
        max_step = pow(max_step, 2);
    }
    penalty = pow(penalty, 2);
    // rows is for series 1, columns is for series 2
    idx_t length = MIN(l2+1, ldiff + 2*window + 1);
    assert(length > 0);
    seq_t * dtw = (seq_t *)malloc(sizeof(seq_t) * length * 2);
    if (!dtw) {
        printf("Error: dtw_distance{{ suffix }} - Cannot allocate memory (size=%zu)\n", length*2);
        return 0;
    }
    idx_t i;
    idx_t j;
    {%- if "ndim" in suffix %}
    idx_t i_idx;
    idx_t j_idx;
    {%- endif %}
    for (j=0; j<length*2; j++) {
        dtw[j] = INFINITY;
    }
    // Deal with psi-relaxation in first row
    for (i=0; i<settings->psi_2b + 1; i++) {
        dtw[i] = 0;
    }
    idx_t skip = 0;
    idx_t skipp = 0;
    int i0 = 1;
    int i1 = 0;
    idx_t minj;
    idx_t maxj;
    idx_t curidx = 0;
    idx_t dl_window = dl + window - 1;
    idx_t ldiff_window = window;
    if (l2 > l1) {
        ldiff_window += ldiff;
    }
    seq_t minv;
    seq_t d;
    seq_t tempv;
    seq_t psi_shortest = INFINITY;
    // keepRunning = 1;
    for (i=0; i<l1; i++) {
        // if (!keepRunning){  // not compatible with OMP
        //     free(dtw);
        //     printf("Stop computing DTW...\n");
        //     return INFINITY;
        // }
        {%- if "ndim" in suffix %}
        i_idx = i * ndim;
        {%- endif %}
        // maxj = i;
        // if (maxj > dl_window) {
        //     maxj -= dl_window;
        // } else {
        //     maxj = 0;
        // }
        maxj = (i - dl_window) * (i > dl_window);
        // No risk for overflow/modulo because we also need to store dtw of size
        // MIN(l2+1, ldiff + 2*window + 1) ?
        minj = i + ldiff_window;
        if (minj > l2) {
            minj = l2;
        }
        skipp = skip;
        skip = maxj;
        i0 = 1 - i0;
        i1 = 1 - i1;
        // Reset new line i1
        for (j=0; j<length; j++) {
            dtw[length * i1 + j] = INFINITY;
        }
        // if (length == l2 + 1) {
        //     skip = 0;
        // }
        skip = skip * (length != l2 + 1);
        // PrunedDTW
        if (sc > maxj) {
            #ifdef DTWDEBUG
            printf("correct maxj to sc: %zu -> %zu (saved %zu computations)\n", maxj, sc, sc-maxj);
            #endif
            maxj = sc;
        }
        smaller_found = false;
        ec_next = i;
        // Deal with psi-relaxation in first column
        if (settings->psi_1b != 0 && maxj == 0 && i < settings->psi_1b) {
            dtw[i1*length + 0] = 0;
        }
        #ifdef DTWDEBUG
        printf("i=%zu, maxj=%zu, minj=%zu\n", i, maxj, minj);
        #endif
        for (j=maxj; j<minj; j++) {
            {%- if "ndim" in suffix %}
            j_idx = j * ndim;
            {%- endif %}
            #ifdef DTWDEBUG
            printf("ri=%zu,ci=%zu, s1[i] = s1[%zu] = %f , s2[j] = s2[%zu] = %f\n", i, j, i, s1[i], j, s2[j]);
            #endif
            {%- if "ndim" in suffix %}
            d = 0;
            for (int d_i=0; d_i<ndim; d_i++) {
                d += SEDIST(s1[i_idx + d_i], s2[j_idx + d_i]);
            }
            {%- if "euclidean" == inner_dist %}
            d = sqrt(d);
            {%- endif %}
            {%- else %}{# ndim not in suffix #}
            {%- if "euclidean" == inner_dist %}
            d = fabs(s1[i] - s2[j]);
            {%- else %}{# inner_dist == "squared euclidean" #}
            d = SEDIST(s1[i], s2[j]);
            {%- endif %}
            {%- endif %}
            if (d > max_step) {
                // Let the value be INFINITY as initialized
                continue;
            }
            curidx = i0 * length + j - skipp;
            minv = dtw[curidx];
            curidx += 1;
            tempv = dtw[curidx] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            curidx = i1 * length + j - skip;
            tempv = dtw[curidx] + penalty;
            if (tempv < minv) {
                minv = tempv;
            }
            #ifdef DTWDEBUG
            printf("d = %f, minv = %f\n", d, minv);
            #endif
            curidx += 1;
            dtw[curidx] = d + minv;
            #ifdef DTWDEBUG
            printf("%zu, %zu, %zu\n",i0*length + j - skipp,i0*length + j + 1 - skipp,i1*length + j - skip);
            printf("%f, %f, %f\n",dtw[i0*length + j - skipp],dtw[i0*length + j + 1 - skipp],dtw[i1*length + j - skip]);
            printf("i=%zu, j=%zu, d=%f, skip=%zu, skipp=%zu\n",i,j,d,skip,skipp);
            #endif
            // PrunedDTW
            if (dtw[curidx] > max_dist) {
                #ifdef DTWDEBUG
                printf("dtw[%zu] = %f > %f\n", curidx, dtw[curidx], max_dist);
                #endif
                if (!smaller_found) {
                    sc = j + 1;
                }
                if (j >= ec) {
                    #ifdef DTWDEBUG
                    printf("Break because of pruning with j=%zu, ec=%zu (saved %zu computations)\n", j, ec, minj-j);
                    #endif
                    break;
                }
            } else {
                smaller_found = true;
                ec_next = j + 1;
            }
        }
        ec = ec_next;
        // Deal with Psi-relaxation in last column
        if (settings->psi_1e != 0 && minj == l2 && l1 - 1 - i <= settings->psi_1e) {
            assert(!(settings->window == 0 || settings->window == l2) || (i1 + 1)*length - 1 == curidx);
            if (dtw[curidx] < psi_shortest) {
                // curidx is the last value
                psi_shortest = dtw[curidx];
            }
        }
        #ifdef DTWDEBUG
        dtw_print_twoline(dtw, l1, l2, length, i0, i1, skip, skipp, maxj, minj);
        #endif
    }
    if (window - 1 < 0) {
        l2 += window - 1;
    }
    {%- if "euclidean" == inner_dist %}
    seq_t result = dtw[length * i1 + l2 - skip];
    {%- else %}
    seq_t result = sqrt(dtw[length * i1 + l2 - skip]);
    {%- endif %}
    // Deal with psi-relaxation in the last row
    if (settings->psi_2e != 0) {
        for (i=l2 - skip - settings->psi_2e; i<l2 - skip + 1; i++) { // iterate over vci
            if (dtw[i1*length + i] < psi_shortest) {
                psi_shortest = dtw[i1*length + i];
            }
        }
        {%- if "euclidean" == inner_dist %}
        result = psi_shortest;
        {%- else %}
        result = sqrt(psi_shortest);
        {%- endif %}
    }
    free(dtw);
    // signal(SIGINT, SIG_DFL);  // not compatible with OMP
    if (settings->max_dist !=0 && result > settings->max_dist) {
        // DTWPruned keeps the last value larger than max_dist. Correct for this.
        result = INFINITY;
    }
    return result;
}

