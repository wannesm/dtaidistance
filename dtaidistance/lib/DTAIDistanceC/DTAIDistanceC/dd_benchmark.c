//
//  benchmark.c
//  DTAIDistanceC
//
//  Copyright © 2020 Wannes Meert.
//  Apache License, Version 2.0, see LICENSE for details.
//

#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <string.h>

#include "dd_dtw.h"
#include "dd_dtw_openmp.h"


void benchmark1(void);
void benchmark2(void);
void benchmark3(void);
void benchmark4(void);
void benchmark5(void);
void benchmark6(void);
void benchmark7(void);
void benchmark8(void);
void benchmark9(void);
void benchmark10(void);
void benchmark11(void);
void benchmark12_subsequence(void);
void benchmark13(void);


void benchmark1() {
//    int size=10000;
//    double ra1[size], ra2[size];
//    int i;
//    for (i=0; i<size; i++) {
//       ra1[i] = rand() % 10;
//       ra2[i] = rand() % 10;
//    }
    
    int size=4;
    double ra1[] = {1., 2, 1, 3};
    double ra2[] = {3., 4, 3, 0};
   
    DTWSettings settings = dtw_settings_default();
//    double d = dtw_distance(ra1, size, ra2, size, &settings);
    settings.window=2;
    double d = lb_keogh(ra1, size, ra2, size, &settings);
   
    printf("... done\n");
    printf("DTW = %f\n", d);
}

void benchmark2() {
    if (is_openmp_supported()) {
        printf("OpenMP is supported\n");
    } else {
        printf("OpenMP is not supported\n");
    }
    int n=1000;
    double s1[] = {0., 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0., 1, 2, 0, 0, 0, 0, 0, 0};
    double s3[] = {1., 2, 0, 0, 0, 0, 0, 1, 1};
    double s4[] = {0., 0, 1, 2, 1, 0, 1, 0, 0};
    double s5[] = {0., 1, 2, 0, 0, 0, 0, 0, 0};
    double s6[] = {1., 2, 0, 0, 0, 0, 0, 1, 1};
    double *s[6*n];
    for (int i=0; i<n; i++) {
        s[i*6+0] = s1;
        s[i*6+1] = s2;
        s[i*6+2] = s3;
        s[i*6+3] = s4;
        s[i*6+4] = s5;
        s[i*6+5] = s6;
    }
    idx_t lengths[6*n];
    for (int i=0; i<6*n; i++) {
        lengths[i] = 9;
    }
    idx_t rl = 6*n*(6*n - 1) / 2;
    printf("Creating result array of size %zu\n", rl);
    double *result = (double *)malloc(sizeof(double) * rl);
    if (!result) {
        printf("Error: benchmark - Cannot allocate memory for result (size=%zu)\n", rl);
        return;
    }
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = dtw_block_empty();
    dtw_distances_ptrs_parallel(s, 6*n, lengths, result, &block, &settings);
//    for (int i=0; i<rl; i++) {
//        printf("%.2f ", result[i]);
//    }
//    printf("\n");
    free(result);
}

void benchmark3() {
    double v = INFINITY;
    printf("v = %f\n", v);
    v = v + 1;
    printf("v + 1 = %f\n", v);
    
    double f = INFINITY;
    uint64_t fn; memcpy(&fn, &f, sizeof f);
    printf("INFINITY:   %f %" PRIx64 "\n", f, fn);
}

void benchmark4() {
//    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0}; int l1 = 9;
//    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0}; int l2 = 9;
//    double s1[] = {0., 0., 1., 2., 1., 0., 1., 0., 0., 2., 1., 0., 0.}; int l1 = 13;
//    double s2[] = {0., 1., 2., 3., 1., 0., 0., 0., 2., 1., 0., 0., 0.}; int l2 = 13;
    double s1[] = {2.1, 4.1, 5.1}; int l1 = 3;
    double s2[] = {1.1, 2.1, 3.1, 4.1, 5.1}; int l2 = 5;
    DTWSettings settings = dtw_settings_default();
    settings.use_pruning = true;
    settings.inner_dist = 0;
    settings.psi_2b = l2;
    settings.psi_2e = l2;
//    dtw_settings_set_psi(2, &settings);
//    double d = dtw_distance(s1, 9, s2, 9, &settings);
    idx_t wps_length = dtw_settings_wps_length(l1, l2, &settings);
    seq_t wps[wps_length];
    double d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, false, &settings);
    printf("d=%f\n", d);
    
    idx_t i1[l1+l2];
    idx_t i2[l1+l2];
    for (idx_t i=0; i<(l1+l2); i++) {i1[i]=0; i2[i]=0;}
    dtw_best_path_isclose(wps, i1, i2, l1, l2, /*rtol=*/1e-05, /*atol=*/1e-08, &settings);
    printf("[");
    for (idx_t i=0; i<(l1+l2); i++) {
        printf("(%zu,%zu)", i1[i], i2[i]);
    }
    printf("]\n");
}

void benchmark5() {
    seq_t s1[] = {0., 0, 1, 2, 1, 0, 1, 0};
    seq_t s2[] = {0., 1, 2, 0, 0, 0, 0, 0};
    idx_t l1 = 4;
    idx_t l2 = 4;
    int ndim = 2;
    DTWSettings settings = dtw_settings_default();
    idx_t wps_length = dtw_settings_wps_length(l1, l2, &settings);
    printf("wps_length=%zu\n", wps_length);
    seq_t wps[wps_length];
//    for (idx_t i=0; i<wps_length; i++) {
//        wps[i] = i;
//    }
    seq_t d = dtw_warping_paths_ndim(wps, s1, l1, s2, l2, true, true, true, ndim, &settings);
    printf("d=%f\n", d);
    dtw_print_wps_compact(wps, l1, l2, &settings);
    printf("\n\n");
    dtw_print_wps(wps, l1, l2, &settings);
    idx_t i1[l1+l2];
    idx_t i2[l1+l2];
    for (idx_t i=0; i<(l1+l2); i++) {i1[i]=0; i2[i]=0;}
    dtw_best_path(wps, i1, i2, l1, l2, &settings);
    printf("[");
    for (idx_t i=0; i<(l1+l2); i++) {
        printf("(%zu,%zu)", i1[i], i2[i]);
    }
    printf("]\n");
    
    seq_t full[(l1+1)*(l2+1)];
    dtw_expand_wps(wps, full, l1, l2, &settings);
    for (idx_t i=0; i<(l1+1); i++) {
        printf("[ ");
        for (idx_t j=0; j<(l2+1); j++) {
            printf("%7.3f ", full[i*(l2+1)+j]);
        }
        printf("]\n");
    }
}

void benchmark6() {
    double s1[] = {0, 0, 0, 1, 2, 1, 0,  1, 0, 0};
    double s2[] = {0, 0, 2, 1, 0, 1, 0, .5, 0, 0};
    DTWSettings settings = dtw_settings_default();
    seq_t d = dtw_distance_ndim(s1, 5, s2, 5, 2, &settings);
    printf("d=%f\n", d);
}

void benchmark7() {
    double s[] = {
        0., 0, 1, 2, 1, 0, 1, 0,
        0., 1, 2, 0, 0, 0, 0, 0,
        1., 2, 0, 0, 0, 0, 0, 1,
        0., 0, 1, 2, 1, 0, 1, 0,
        0., 1, 2, 0, 0, 0, 0, 0,
        1., 2, 0, 0, 0, 0, 0, 1};
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = dtw_block_empty();
    double result[dtw_distances_length(&block, 6, 6)];
    dtw_distances_ndim_matrix(s, 6, 4, 2, result, &block, &settings);
}

void benchmark8() {
     double s1[] = {0., 0, 1, 2, 1, 0, 1, 0, 0};
         double s2[] = {0., 1, 2, 0, 0, 0, 0, 0, 0};
         double s3[] = {1., 2, 0, 0, 0, 0, 0, 1, 1};
         double s4[] = {0., 0, 1, 2, 1, 0, 1, 0, 0};
         double s5[] = {0., 1, 2, 0, 0, 0, 0, 0, 0};
         double s6[] = {1., 2, 0, 0, 0, 0, 0, 1, 1};
         double *s[] = {s1, s2, s3, s4, s5, s6};
         idx_t lengths[] = {9, 9, 9, 9, 9, 9};
         double result[5];
         DTWSettings settings = dtw_settings_default();
         DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5};
         for (int j=0; j<1000; j++) {
//             printf("---START---\n");
             dtw_distances_ptrs_parallel(s, 6, lengths, result, &block, &settings);
//             for (int i=0; i<5; i++) {
//                 printf("%.4f ", result[i]);
//             }
//             printf("\n");
//             cr_assert_float_eq(result[0], 1.41421356, 0.001);
//             cr_assert_float_eq(result[1], 0.00000000, 0.001);
//             cr_assert_float_eq(result[2], 2.23606798, 0.001);
//             cr_assert_float_eq(result[3], 1.73205081, 0.001);
//             cr_assert_float_eq(result[4], 1.41421356, 0.001);
         }
}

void benchmark9() {
//    double s[] = {
//        0.5, 1, 2, 3, 2.0, 2.1, 1.0, 0, 0, 0, // Row 0
//        0.4, 0, 1, 1.5, 1.9, 2.0, 0.9, 1, 0, 0 // Row 1
//    };
    //    double exp_avg[] = {0.3, 1.1666666666666667, 1.95, 2.5, 2.0, 2.05, 0.9666666666666667, 0.0, 0.0, 0.0};
//    int ndim = 1;
    double s[] = {
        0., 0, 1, 2, 1, 0, 1, 0,
        0., 1, 2, 0, 0, 0, 0, 0,
        1., 2, 0, 0, 0, 0, 0, 1,
        0., 0, 1, 2, 1, 0, 1, 0,
        0., 1, 2, 0, 0, 0, 0, 0,
        1., 2, 0, 0, 0, 0, 0, 1
    };
    double exp_avg[] = {0.33333333, 1., 0.66666667, 1.66666667, 0.6, 0., 0.33333333, 0.33333333};
    int ndim = 2;
    idx_t nb_cols = 4;
    idx_t nb_rows = 6;
    seq_t c[nb_cols * ndim];
    for (idx_t i=0; i<(nb_cols*ndim); i++) { // Copy first series
        c[i] = s[i];
    }
//    bit_array(mask, nb_rows)
    ba_t mask[bit_bytes(nb_rows)];
    for (int i=0; i<nb_rows; i++) {
        bit_set(mask, i);
    }
    DTWSettings settings = dtw_settings_default();
    
    dtw_dba_matrix(s, nb_rows, nb_cols, c, nb_cols, mask, 0, ndim, &settings);
    
    printf("Computed avg:\n");
    for (int i=0; i<(nb_cols*ndim); i++) {
        printf("%f ", c[i]);
    }
    printf("\n");
}

void benchmark10() {
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0};
    int l1 = 4;
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0};
    int l2 = 4;
    idx_t i1s[] = {3, 2, 2, 1, 0};
    idx_t i2s[] = {3, 2, 1, 0, 0};
    idx_t ils = 5;
    DTWSettings settings = dtw_settings_default();
//    settings.window = 5;
//    dtw_settings_set_psi(2, &settings);
    int ndim = 2;
    
    idx_t *i1 = (idx_t *)malloc((l1 + l2) * sizeof(idx_t));
    idx_t *i2 = (idx_t *)malloc((l1 + l2) * sizeof(idx_t));
    idx_t length_i;
    seq_t d;

    d = dtw_distance_ndim(s1, l1, s2, l2, ndim, &settings);
    printf("d =. %.2f\n", d);
    
    d = dtw_warping_path_ndim(s1, l1, s2, l2, i1, i2, &length_i, ndim, &settings);
    
    printf("d = %.2f\n", d);
    printf("path[:%zu] = [", length_i);
    for (idx_t i=length_i-1; i>=0; i--) {
        printf("(%zd, %zd), ", i1[i], i2[i]);
    }
    printf("]\n");
    
    DTWWps p = dtw_wps_parts(l1, l2, &settings);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * p.length);
    d = dtw_warping_paths_ndim(wps, s1, l1, s2, l2, true, true, true, ndim, &settings);
    dtw_print_wps_compact(wps, l1, l2, &settings);
    
}

void benchmark11() {
    double s1[] = {0.00,0.48,0.84,1.00,0.91,0.60,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,-0.28,0.22,
                   0.66,0.94,0.99,0.80,0.41,-0.08,-0.54,-0.88,-1.00,-0.88,-0.54,-0.07,0.42,
                   0.80,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34,0.15,0.61};
    double s2[] = {-0.84,-0.48,0.00,0.48,0.84,1.00,0.91,0.60,0.14,-0.18,-0.76,-0.98,-0.99,-0.71,
                   -0.28,0.22,0.66,0.70,0.99,0.80,0.41,-0.08,-0.54,-1.02,-1.00,-0.88,-0.54,
                   -0.07,0.42,0.80,0.99,1.10,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34};
    DTWSettings settings = dtw_settings_default();
    dtw_settings_set_psi(2, &settings);
    settings.window = 25;
    
    double dd = dtw_distance(s2, 40, s1, 40, &settings);
    printf("dd=%f\n", dd);
    
    idx_t l1 = 40;
    idx_t l2 = 40;
    idx_t wps_length = dtw_settings_wps_length(l1, l2, &settings);
    printf("wps_length=%zu\n", wps_length);
    seq_t wps[wps_length];
    seq_t d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, true, &settings);
    printf("d=%f\n", d);
    dtw_print_wps_compact(wps, l1, l2, &settings);
    printf("\n\n");
    dtw_print_wps(wps, l1, l2, &settings);
    idx_t i1[l1+l2];
    idx_t i2[l1+l2];
    for (idx_t i=0; i<(l1+l2); i++) {i1[i]=0; i2[i]=0;}
    dtw_best_path_prob(wps, i1, i2, l1, l2, d/l1, &settings);
    printf("best_path = [");
    for (idx_t i=0; i<(l1+l2); i++) {
        printf("(%zu,%zu)", i1[i], i2[i]);
    }
    printf("]\n");
}


void benchmark12_subsequence() {
    seq_t s1[] = {1, 2, 0};  // query
    seq_t s2[] = {1, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0};
    idx_t l1 = 3;
    idx_t l2 = 11;
    
    DTWSettings settings = dtw_settings_default();
    settings.window = 0;
    settings.psi_1b = 0;
    settings.psi_1e = 0;
    settings.psi_2b = l2;
    settings.psi_2e = l2;
    settings.penalty = 0.1;
    settings.use_pruning = false;
    idx_t wps_length = dtw_settings_wps_length(l1, l2, &settings);
    printf("wps_length=%zu\n", wps_length);
    seq_t wps[wps_length];
//    for (idx_t i=0; i<wps_length; i++) {
//        wps[i] = i;
//    }
    printf("window=%zu\n", settings.window);
    seq_t d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, false, &settings);
    printf("d=%f\n", d);
    printf("window=%zu\n", settings.window);
    printf("Compact:\n");
    dtw_print_wps_compact(wps, l1, l2, &settings);
    printf("\n\n");
    dtw_print_wps(wps, l1, l2, &settings);
    idx_t i1[l1+l2];
    idx_t i2[l1+l2];
    for (idx_t i=0; i<(l1+l2); i++) {i1[i]=0; i2[i]=0;}
    dtw_best_path(wps, i1, i2, l1, l2, &settings);
    printf("[");
    for (idx_t i=0; i<(l1+l2); i++) {
        printf("(%zu,%zu)", i1[i], i2[i]);
    }
    printf("]\n");
    
    seq_t full[(l1+1)*(l2+1)];
    dtw_expand_wps(wps, full, l1, l2, &settings);
    for (idx_t i=0; i<(l1+1); i++) {
        printf("[ ");
        for (idx_t j=0; j<(l2+1); j++) {
            printf("%7.3f ", full[i*(l2+1)+j]);
        }
        printf("]\n");
    }
}


void benchmark13() {
    double s1[] = {-0.86271501, -1.32160597, -1.2307838, -0.97743775, -0.88183547, -0.71453147, -0.70975136, -0.65238999, -0.48508599, -0.40860416, -0.5567877, -0.39904393, -0.51854679, -0.51854679, -0.23652005, -0.21261948, 0.16978966, 0.21281068, 0.6573613, 1.28355626, 1.88585065, 1.565583, 1.40305912, 1.64206483, 1.8667302};
    double s2[] = {-0.87446789, 0.50009064, -1.43396157, 0.52081263, 1.28752619};
    DTWSettings settings = dtw_settings_default();
    settings.psi_1b = 0;
    settings.psi_1e = 0;
    settings.psi_2b = 5; // len(s2)
    settings.psi_2e = 5; // len(s2)
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * (25+1)*(5+1));
    double d = dtw_warping_paths(wps, s1, 25, s2, 5, true, true, true, &settings);
//    printf("d = %7.3f\n", d);
//    for (int ri=0; ri<26; ri++) {
//        for (int ci=0; ci<6; ci++) {
//            printf("%7.3f, ", wps[ri*6+ci]);
//        }
//        printf("\n");
//    }
    free(wps);
}

void benchmark_affinity() {
    dtw_printprecision_set(3);
    double s[] = {0, -1, -1, 0, 1, 2, 1};
    idx_t l1 = 7;
    idx_t l2 = 7;
    DTWSettings settings = dtw_settings_default();
    settings.window = 2;
    settings.penalty = 1;
    idx_t wps_width = dtw_settings_wps_width(l1, l2, &settings);
    printf("wps_width=%zu\n", wps_width);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * (l2+1)*wps_width);
    seq_t tau = 0.36787944117144233;
    seq_t delta = -0.7357588823428847;
    seq_t delta_factor = 0.5;
    seq_t gamma = 1;
    double d = dtw_warping_paths_affinity(wps, s, 7, s, 7, true, false,
                                          /*psi_neg=*/true,
                                          /*only_triu=*/false,
                                          gamma, tau, delta, delta_factor, &settings);
    dtw_print_wps_compact(wps, 7, 7, &settings);
    
    idx_t i1[l1+l2];
    idx_t i2[l1+l2];
    for (idx_t i=0; i<(l1+l2); i++) {i1[i]=0; i2[i]=0;}
    idx_t result = dtw_best_path_affinity(wps, i1, i2, l1, l2, l1-3, l2-2, &settings);
    printf("result=%zu\n", result);
    printf("[");
    for (idx_t i=0; i<(l1+l2); i++) {
        printf("(%zu,%zu)", i1[i], i2[i]);
    }
    printf("]\n");
    
    dtw_print_wps(wps, l1, l2, &settings);
    DTWWps p = dtw_wps_parts(l1, l2, &settings);
    
    printf("Slice:\n");
    idx_t rb = 4;
    idx_t re = 7;
    idx_t cb = 3;
    idx_t ce = 6;
    seq_t * wps_slice = (seq_t *)malloc(sizeof(seq_t) * (re-rb)*(ce-cb));
    for (idx_t i=0; i<(re-rb)*(ce-cb); i++) {
        wps_slice[i] = -INFINITY;
    }

    dtw_expand_wps_slice_affinity(wps, wps_slice, l1, l2, rb, re, cb, ce, &settings);

    idx_t wpsi = 0;
    for (idx_t r=0; r<(re-rb); r++) {
        printf("[ ");
        for (idx_t c=0; c<(ce-cb); c++) {
            printf("%.2f ", wps_slice[wpsi]);
            wpsi++;
        }
        printf("]\n");
    }
    
//    dtw_wps_negativize(&p, wps, 2, 5);
//    dtw_wps_positivize(&p, wps, 3, 4);
    
    
//    idx_t r, c, wps_i;
//    r = l1-3; c = l2-2;
//    wps_i = dtw_wps_loc(&p, r, c, l1, l2);
//    printf("wps_full[%zu,%zu] = wps[%zu] = %.3f\n", r, c, wps_i, wps[wps_i]);
    
//    idx_t maxr, maxc;
//    idx_t maxidx = dtw_wps_max(&p, wps, &maxr, &maxc, l1, l2);
//    printf("Max = %.3f @ [%zu]=[%zu,%zu]\n", wps[maxidx], maxidx, maxr, maxc);
    
    printf("Negativize\n");
    dtw_wps_negativize(&p, wps, l1, l2, 4, 6, 4, 5);
    dtw_print_wps(wps, l1, l2, &settings);
    dtw_print_wps_compact(wps, l1, l2, &settings);
//    maxidx = dtw_wps_max(&p, wps, &maxr, &maxc, l1, l2);
//    printf("Max = %.3f @ [%zu]=[%zu,%zu]\n", wps[maxidx], maxidx, maxr, maxc);
    
    free(wps);
    printf("d = %.2f\n", d);
    dtw_printprecision_reset();
}

void wps_test(void) {
    dtw_printprecision_set(0);
    
    idx_t l1 = 8065;
    idx_t l2 = 8065;
    idx_t idx;
    
    DTWSettings settings = dtw_settings_default();
    settings.window = 50;
    settings.penalty = 0.0018315638888734178;
    idx_t wps_width = dtw_settings_wps_width(l1, l2, &settings);
    printf("wps_width=%zu\n", wps_width);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * (l2+1)*wps_width);
    seq_t * series1 = (seq_t *)malloc(sizeof(seq_t) * l1);
    seq_t * series2 = (seq_t *)malloc(sizeof(seq_t) * l2);
    
    FILE *in_file;
    double number;

    // read series
    in_file = fopen("/Users/wannes/Projects/Research/2016-DTW/repo_dtw/tests/rsrc/series1.txt", "r");
    if (in_file == NULL) {
        printf("Can't open file for reading.\n");
        return;
    }
    idx = 0;
    for (idx_t i=0; i<l1+1; i++) {
        fscanf(in_file, "%lf", &number);
        series1[idx] = number;
        idx++;
    }
    in_file = fopen("/Users/wannes/Projects/Research/2016-DTW/repo_dtw/tests/rsrc/series2.txt", "r");
    if (in_file == NULL) {
        printf("Can't open file for reading.\n");
        return;
    }
    idx = 0;
    for (idx_t i=0; i<l1+1; i++) {
        fscanf(in_file, "%lf", &number);
        series2[idx] = number;
        idx++;
    }
    
    // compute wps
    seq_t tau = 0.01831563888873418;
    seq_t delta = -0.03663127777746836;
    seq_t delta_factor = 0.9;
    seq_t gamma = 0.0008575903363340125;
    double d = dtw_warping_paths_affinity(wps, series1, l1, series2, l2, true, false,
                                          /*psi_neg=*/true,
                                          /*only_triu=*/false,
                                          gamma, tau, delta, delta_factor, &settings);
    printf("d=%.2f\n", d);
    
    // read wps
    in_file = fopen("/Users/wannes/Projects/Research/2016-DTW/repo_dtw/tests/rsrc/wps.txt", "r");
    if (in_file == NULL) {
        printf("Can't open file for reading.\n");
        return;
    }
    idx = 0;
    for (idx_t i=0; i<l1+1; i++) {
        for (idx_t j=0; j<wps_width; j++) {
            fscanf(in_file, "%lf", &number);
            assert(wps[idx] == number);
            idx++;
        }
    }
    
//    dtw_print_wps(wps, l1, l2, &settings);
    DTWWps p = dtw_wps_parts(l1, l2, &settings);
    
    printf("Negativize\n");
    dtw_wps_negativize(&p, wps, l1, l2, 1, 7930, 1, 7943);
    
    printf("Slice:\n");
    idx_t rb = 7928; //7900;
    idx_t re = rb+10; //7950;
    idx_t cb = rb-60;//rb; //7900;
    idx_t ce = rb+60;//re; //7950;
    seq_t * wps_slice = (seq_t *)malloc(sizeof(seq_t) * (re-rb)*(ce-cb));
    for (idx_t i=0; i<(re-rb)*(ce-cb); i++) {
        wps_slice[i] = -INFINITY;
    }
    dtw_expand_wps_slice_affinity(wps, wps_slice, l1, l2, rb, re, cb, ce, &settings);
    idx_t wpsi = 0;
    for (idx_t r=0; r<(re-rb); r++) {
        printf("[ ");
        for (idx_t c=0; c<(ce-cb); c++) {
            dtw_print_nb(wps_slice[wpsi]);
            wpsi++;
        }
        printf("]\n");
    }
    
    wpsi = rb*p.width;
    printf("wps (wpsi=%zu):\n", wpsi);
//    for (idx_t r=0; r<(re-rb); r++) {
//        printf("[ ");
//        for (idx_t c=0; c<p.width; c++) {
//            dtw_print_nb(wps[wpsi]);
//            wpsi++;
//        }
//        printf("]\n");
//    }
    
    
    free(wps);
}


int main(int argc, const char * argv[]) {
    printf("Benchmarking ...\n");
    time_t start_t, end_t;
    struct timespec start, end;
    double diff_t, diff_t2;
    time(&start_t);
    clock_gettime(CLOCK_REALTIME, &start);
    
//    benchmark1();
//    benchmark2();
//    benchmark3();
    benchmark4();
//    benchmark5();
//    benchmark6();
//    benchmark7();
//    benchmark8();
//    benchmark9();
//    benchmark10();
//    benchmark11();
//    benchmark12_subsequence();
//    benchmark13();
//    benchmark_affinity();
//    wps_test();
    
    time(&end_t);
    clock_gettime(CLOCK_REALTIME, &end);
    diff_t = difftime(end_t, start_t);
    diff_t2 = ((double)end.tv_sec*1e9 + end.tv_nsec) - ((double)start.tv_sec*1e9 + start.tv_nsec);
    printf("Execution time = %f sec / %f ms\n", diff_t, diff_t2/1000000);
    
    return 0;
}


