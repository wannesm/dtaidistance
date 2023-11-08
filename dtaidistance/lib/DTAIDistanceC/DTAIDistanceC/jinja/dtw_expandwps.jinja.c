{%- if "affinity" in suffix -%}
{%- set infinity="-INFINITY" -%}
{%- else -%}
{%- set infinity="INFINITY" -%}
{%- endif -%}

/*!
 Expand the compact wps datastructure to a full `(l1+1)*(l2+1)` sized matrix.
 */
void dtw_expand_wps{{suffix}}(seq_t *wps, seq_t *full,
                        idx_t l1, idx_t l2, DTWSettings *settings) {
    dtw_expand_wps_slice{{suffix}}(wps, full, l1, l2, 0, l1+1, 0, l2+1, settings);
}

/*!
 Expand the compact wps datastructure to a full `(re-rb)*(ce-cb)` sized matrix that
 represents the slice `[rb:re,cb:ce]` of the full matrix.
 
 @param wps Compact warping paths matrix
 @param full Sequence of length `(re-rb)*(ce-cb)`
        Will be filled with values.
 @param l1 Length of series 1
 @param l2 Length of series 2
 @param rb Start of slice row (0 <= rb <= l1+1)
 @param re End of slice row (0 <= rb <= l1+1)
 @param cb Start of slice column (0 <= rc <= l2+1)
 @param ce End of slice column (0 <= rc <= l2+1)
 @param settings DTWSetting object
 */
void dtw_expand_wps_slice{{suffix}}(seq_t *wps, seq_t *full,
                    idx_t l1, idx_t l2,
                    idx_t rb, idx_t re, idx_t cb, idx_t ce,
                    DTWSettings *settings) {
    DTWWps p = dtw_wps_parts(l1, l2, settings);

    idx_t ri, ci, min_ci, max_ci, wpsi, wpsi_start;
    idx_t rbs = 0;
    if (rb > 0) { rbs = rb - 1; }
    idx_t res = 0;
    if (re > 0) { res = re - 1; }
    idx_t cbs = 0;
    if (cb > 0) { cbs = cb - 1; }
    idx_t ces = 0;
    if (ce > 0) { ces = ce - 1; }
    idx_t fwidth = ce - cb;

    for (idx_t i=0; i<(re-rb)*(ce-cb); i++) {
        full[i] = {{infinity}};
    }

    // Top row: ri = -1
    if (rb == 0 && cb == 0) {
        full[0] = wps[0];
    }
    if (rb == 0) {
        wpsi = 1 + cbs;
        for (ci=cbs; ci<MIN3(ces, p.width - 1, l2); ci++) {
            full[wpsi-cbs] = wps[wpsi];
            wpsi++;
        }
    }

    // A. Rows: 0 <= ri < min(overlap_left_ri, overlap_right_ri)
    if (rbs < p.ri1) {
        min_ci = 0;
        max_ci = p.window + p.ldiffc; // ri < overlap_right_i
        max_ci += rbs;
        for (ri=rbs; ri<MIN(res, p.ri1); ri++) {
            if (cbs == 0) {
                full[fwidth*(ri + 1)] = wps[p.width*(ri + 1)];
            }
            if (cbs <= min_ci) {
                wpsi = 1;
            } else {
                wpsi = 1 + (cbs - min_ci);
            }
            for (ci=MAX(cbs, min_ci); ci<MIN(ces, max_ci); ci++) {
                full[(ri+1-rb)*fwidth + ci + 1 - cb] = wps[(ri+1)*p.width + wpsi];
                wpsi++;
            }
            max_ci++;
        }
    }

    // B. Rows: min(overlap_left_ri, overlap_right_ri) <= ri < overlap_left_ri
    min_ci = cbs;
    max_ci = MIN(ces, l2); // ri >= overlap_right_i
    if (rbs < p.ri2) {
        for (ri=MAX(rbs, p.ri1); ri<MIN(res, p.ri2); ri++) {
            if (cbs == 0) {
                full[fwidth*(ri + 1)] = wps[p.width*(ri + 1)];
            }
            if (cbs <= min_ci) {
                wpsi = 1;
            } else {
                wpsi = 1 + (cbs - min_ci);
            }
            for (ci=MAX(cbs, min_ci); ci<MIN(ces, max_ci); ci++) {
                full[(ri+1-rb)*fwidth + ci + 1 - cb] = wps[(ri+1)*p.width + wpsi];
                wpsi++;
            }
        }
    }

    // C. Rows: overlap_left_ri <= ri < MAX(parts.overlap_left_ri, parts.overlap_right_ri)
    min_ci = 1;
    max_ci = 1 + 2 * p.window - 1 + p.ldiff;
    if (rbs < p.ri3) {
        // if (rbs > p.ri2) {
        //     min_ci += rbs - p.ri2;
        //     max_ci += rbs - p.ri2;
        // }
        for (ri=MAX(rbs, p.ri2); ri<MIN(res, p.ri3); ri++) {
            if (cbs == 0) {
                full[(ri+1)*fwidth + min_ci] = wps[(ri+1)*p.width + 0];
            }
            if (cbs <= min_ci) {
                wpsi = 1;
            } else {
                wpsi = 1 + (cbs - min_ci);
            }
            for (ci=MAX(cbs, min_ci); ci<MIN(ces, max_ci); ci++) {
                full[(ri+1-rb)*fwidth + ci + 1 - cb] = wps[(ri+1)*p.width + wpsi];
                wpsi++;
            }
            min_ci++;
            max_ci++;
        }
    }

    // D. Rows: MAX(overlap_left_ri, overlap_right_ri) < ri <= l1
    min_ci = p.ri3 + 1 - p.window - p.ldiff;
    wpsi_start = 2;
    if (p.ri2 == p.ri3) {
        // C is skipped
        wpsi_start = min_ci + 1;
    } else {
        min_ci = 1 + p.ri3 - p.ri2;
    }
    // if (rbs > p.ri3) {
    //     min_ci += rbs - p.ri3;
    //     wpsi_start += rbs - p.ri3;
    // }
    for (ri=MAX(rbs, p.ri3); ri<MIN(res, l1); ri++) {
        if (cbs <= min_ci) {
            wpsi = wpsi_start;
        } else {
            wpsi = wpsi_start + (cbs - min_ci);
        }
        for (ci=MAX(cbs, min_ci); ci<MIN(ces, l2); ci++) {
            full[(ri+1-rb)*fwidth + ci + 1 - cb] = wps[(ri+1)*p.width + wpsi];
            wpsi++;
        }
        min_ci++;
        wpsi_start++;
    }
}
