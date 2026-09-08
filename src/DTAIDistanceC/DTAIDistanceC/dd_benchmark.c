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
#include "dd_loco.h"


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
    
    for (idx_t i=0; i<(l1+l2); i++) {i1[i]=0; i2[i]=0;}
    dtw_best_path_customstart(wps, i1, i2, l1, l2, /*rs=*/3, /*cs=*/2, &settings);
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

void benchmark11b() {
    double s1[] = {2.00000000e+00, 2.00000000e+00, 2.00000000e+00, 2.00000000e+00,
        2.00000000e+00, 2.00000000e+00, 2.00000000e+00, 2.00000000e+00,
        2.00000000e+00, 2.00000000e+00, 1.66586454e-02, 1.18279664e-03,
        5.34785084e-03, 1.83520337e-03, 1.62359278e-02, 3.35703836e-03,
        6.32859316e-03, 9.66213392e-03, 5.49752452e-04, 9.50024391e-03,
        1.22293821e-02, 8.85627835e-03, 1.27947120e-02, 1.30600475e-02,
        7.23919324e-03, 9.50952294e-03, 1.69907835e-02, 4.28076648e-04,
        1.06846177e-02, 1.04737503e-02, 1.63459325e-02, 9.78673992e-03,
        3.02520524e-03, 8.15589446e-03, 1.25039705e-02, 1.06103404e-03,
        9.83433929e-03, 1.66517248e-02, 1.65437612e-02, 2.15865110e-02,
        1.14567510e-02, 2.19864178e-02, 3.26690723e-02, 2.49917912e-02,
        3.76505664e-02, 5.45575871e-02, 7.45688834e-02, 9.46177975e-02,
        1.08188902e-01, 1.40216138e-01, 1.69603695e-01, 2.10885734e-01,
        2.71845652e-01, 3.24146801e-01, 3.76109038e-01, 4.49809249e-01,
        5.23336629e-01, 6.00004162e-01, 6.87279669e-01, 8.15221243e-01,
        1.00268081e+00, 1.25359351e+00, 1.38472283e+00, 1.40040835e+00,
        1.31976820e+00, 1.21015660e+00, 1.08899108e+00, 9.88023391e-01,
        9.07917695e-01, 8.46209268e-01, 7.96154363e-01, 7.53764246e-01,
        7.28284595e-01, 6.84030941e-01, 6.67982709e-01, 6.50753650e-01,
        6.24812141e-01, 6.17615166e-01, 6.15226890e-01, 6.07320889e-01,
        5.90530835e-01, 5.85914346e-01, 5.84093574e-01, 5.86096523e-01,
        5.73854533e-01, 5.66875811e-01, 5.75328631e-01, 5.71739107e-01,
        5.72299116e-01, 5.69736358e-01, 5.72096734e-01, 5.67548067e-01,
        5.55368496e-01, 5.59258799e-01, 5.64453418e-01, 5.56610534e-01,
        5.54883693e-01, 5.57511874e-01, 5.52333891e-01, 5.52111827e-01,
        5.43299929e-01, 5.55368484e-01, 5.48276270e-01, 5.58337092e-01,
        5.54916880e-01, 5.46638430e-01, 5.51415683e-01, 5.49815214e-01,
        5.34835688e-01, 5.37612990e-01, 5.38007643e-01, 5.37461227e-01,
        5.51547082e-01, 5.42623754e-01, 5.32467470e-01, 5.36047386e-01,
        5.37175837e-01, 5.33998213e-01, 5.45062879e-01, 5.43006151e-01,
        5.29077178e-01, 5.40061043e-01, 5.32486910e-01, 5.25930175e-01,
        5.18728780e-01, 5.00319982e-01, 4.82677196e-01, 4.42038600e-01,
        4.03638443e-01, 3.62306329e-01, 3.50516551e-01, 4.17266572e-01,
        5.32266650e-01, 5.26076706e-01, 6.15382956e-01, 6.94597594e-01,
        7.08659194e-01, 6.83027875e-01, 6.38406763e-01, 6.07419830e-01,
        5.81500528e-01, 5.42260167e-01, 5.32713987e-01, 5.29401970e-01,
        5.23952233e-01, 5.36287694e-01, 5.35565256e-01, 5.37215622e-01,
        5.25704791e-01, 5.21554315e-01, 5.23657732e-01, 5.35410267e-01,
        5.24130795e-01, 5.20925680e-01, 5.19679256e-01, 5.32584421e-01,
        5.17390904e-01, 5.21931589e-01, 5.34157165e-01, 5.26714575e-01,
        5.35095562e-01, 5.27222636e-01, 5.16818329e-01, 5.21778505e-01,
        5.31558860e-01, 5.26062367e-01, 5.22994216e-01, 5.17230187e-01,
        5.24277816e-01, 5.18922106e-01, 5.26117109e-01, 5.26818781e-01,
        5.16841751e-01, 5.34308482e-01, 5.25116961e-01, 5.25898774e-01,
        5.19360070e-01, 5.13608556e-01, 4.01157184e-01, 3.50619200e-02,
        3.28716445e-01, 5.15356783e-01, 5.15140095e-01, 5.27097899e-01,
        5.33203288e-01, 5.29719030e-01, 5.21855069e-01, 5.18916184e-01,
        5.15119932e-01, 5.19799062e-01, 5.30328931e-01, 5.19079143e-01,
        5.28337307e-01, 5.19291573e-01, 5.20431261e-01, 5.16597436e-01,
        5.21682277e-01, 5.23297683e-01, 5.18198642e-01, 5.18728704e-01};
    double s2[] = {2.58838998e-03, 1.39763164e-02, 1.86647206e-02, 1.23463504e-02,
        5.34709021e-03, 6.11372966e-03, 1.66037781e-02, 1.42073180e-02,
        1.61513057e-02, 2.78136320e-03, 1.66744951e-02, 1.20467577e-03,
        5.37834838e-03, 1.87825523e-03, 1.62976511e-02, 3.44711332e-03,
        6.46252876e-03, 9.86492246e-03, 8.61645229e-04, 9.98566092e-03,
        1.29902871e-02, 1.00516634e-02, 1.46680645e-02, 1.59767089e-02,
        1.17352065e-02, 1.63527534e-02, 2.72536455e-02, 1.55690527e-02,
        3.26330161e-02, 4.17091261e-02, 5.99608672e-02, 6.95200550e-02,
        8.32531391e-02, 1.13832269e-01, 1.49047602e-01, 1.74199825e-01,
        2.25430130e-01, 2.80546875e-01, 3.34487816e-01, 3.99317233e-01,
        4.55022781e-01, 5.38408447e-01, 6.31043916e-01, 7.18152313e-01,
        8.44509234e-01, 1.07663846e+00, 1.31349120e+00, 1.41753090e+00,
        1.38510810e+00, 1.29056683e+00, 1.16587004e+00, 1.05670904e+00,
        9.80145502e-01, 9.06163955e-01, 8.38966388e-01, 7.97137917e-01,
        7.55828105e-01, 7.15023068e-01, 6.77549521e-01, 6.66684629e-01,
        6.48898226e-01, 6.28073639e-01, 6.08702017e-01, 6.08290248e-01,
        6.03369788e-01, 6.04771939e-01, 5.91687511e-01, 5.81516120e-01,
        5.74606855e-01, 5.72274814e-01, 5.71098909e-01, 5.69184559e-01,
        5.77086575e-01, 5.60090415e-01, 5.66027486e-01, 5.66316207e-01,
        5.54186176e-01, 5.57794890e-01, 5.63828173e-01, 5.62491054e-01,
        5.50855606e-01, 5.50329294e-01, 5.51805155e-01, 5.56515393e-01,
        5.46541779e-01, 5.41501325e-01, 5.51639998e-01, 5.49539057e-01,
        5.51429597e-01, 5.50067506e-01, 5.53519348e-01, 5.49968488e-01,
        5.38705117e-01, 5.43439647e-01, 5.49414438e-01, 5.42294326e-01,
        5.41238554e-01, 5.44491047e-01, 5.39894945e-01, 5.40216151e-01,
        5.31912298e-01, 5.44456688e-01, 5.37810794e-01, 5.48290838e-01,
        5.45264919e-01, 5.37357790e-01, 5.42485157e-01, 5.41215193e-01,
        5.26548013e-01, 5.29620816e-01, 5.30295327e-01, 5.30014217e-01,
        5.44351824e-01, 5.35667610e-01, 5.25738639e-01, 5.29534834e-01,
        5.30869239e-01, 5.27887893e-01, 5.39139763e-01, 5.37261718e-01,
        5.23503420e-01, 5.34650424e-01, 5.27232336e-01, 5.20824957e-01,
        5.13766610e-01, 4.95494903e-01, 4.77983578e-01, 4.37471118e-01,
        3.99192060e-01, 3.57976270e-01, 3.46298293e-01, 4.13155823e-01,
        5.28259335e-01, 5.22168955e-01, 6.11571091e-01, 6.90878115e-01,
        7.05028772e-01, 6.79483337e-01, 6.34945088e-01, 6.04038136e-01,
        5.78196066e-01, 5.39030313e-01, 5.29556235e-01, 5.26313925e-01,
        5.20931604e-01, 5.33332292e-01, 5.32672985e-01, 5.34384475e-01,
        5.22932845e-01, 5.18839728e-01, 5.20998737e-01, 5.32805169e-01,
        5.21577968e-01, 5.18423563e-01, 5.17226348e-01, 5.30179283e-01,
        5.15032150e-01, 5.19617886e-01, 5.31887233e-01, 5.24487181e-01,
        5.32909517e-01, 5.25076798e-01, 5.14711597e-01, 5.19709818e-01,
        5.29527193e-01, 5.24066734e-01, 5.21033664e-01, 5.15303799e-01,
        5.22384705e-01, 5.17061414e-01, 5.24288010e-01, 5.25020475e-01,
        5.15073466e-01, 5.32569471e-01, 5.23406502e-01, 5.24216171e-01,
        5.17713687e-01, 5.15310979e-01, 5.25978357e-01, 5.27834667e-01,
        5.26336987e-01, 5.22096217e-01, 5.13669996e-01, 5.25512734e-01,
        5.15289640e-01, 2.56903170e-01, 5.92523313e-02, 4.36765583e-01,
        5.12288039e-01, 5.18441538e-01, 5.28993766e-01, 2.64496883e-01,
        1.32248441e-01, 6.61242207e-02, 3.30621104e-02, 1.65310552e-02,
        8.26552759e-03, 4.13276380e-03, 2.06638190e-03, 1.03319095e-03};
    DTWSettings settings = dtw_settings_default();
    dtw_settings_set_psi(10, &settings);
//    settings.window = 25;
    idx_t l1 = 200;
    idx_t l2 = 200;
    
    double dd = dtw_distance(s2, l1, s1, l2, &settings);
    printf("dd=%f\n", dd);
    
    idx_t wps_length = dtw_settings_wps_length(l1, l2, &settings);
    printf("wps_length=%zu\n", wps_length);
    seq_t wps[wps_length];
    seq_t d = dtw_warping_paths(wps, s1, l1, s2, l2, true, false, true, &settings);
    printf("d=%f\n", d);
    dtw_print_wps_compact(wps, l1, l2, &settings);
    printf("\n\n");
    dtw_print_wps(wps, l1, l2, &settings);
    idx_t i1[l1+l2];
    idx_t i2[l1+l2];
    for (idx_t i=0; i<(l1+l2); i++) {i1[i]=0; i2[i]=0;}
    dtw_best_path(wps, i1, i2, l1, l2, &settings);
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

void benchmark14() {
    double s1[] = {0.0, 1.0};
    double s2[] = {0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.psi_1e = 1;
//    settings.psi_2e = 1;
    double d = dtw_distance(s1, 2, s2, 2, &settings);
    printf("d = %f\n", d);
}

void benchmark15_subsequence() {
    seq_t s1[] = {1, 2, 0};  // query
    seq_t s2[] = {1, 0, 1, 2, 1, 0, 2, 0, 3, 0, 0};
    idx_t l1 = 3;
    idx_t l2 = 11;
    seq_t s1b[] = {1, 2, 0};
    seq_t s2b[] = {1, 0, 1, 2, 1, 0, 2, 0, 3, 0, 0};
    
    DTWSettings settings = dtw_settings_default();
    settings.window = 0;
    settings.psi_1b = 0;
    settings.psi_1e = 0;
    settings.psi_2b = l2;
    settings.psi_2e = 0;
    settings.penalty = 0.1;
    settings.use_pruning = false;
    idx_t wps_length = dtw_settings_wps_length(l1, l2, &settings);
    printf("wps_length=%zu\n", wps_length);
    seq_t wps[wps_length];
    seq_t wpsb[wps_length];
//    for (idx_t i=0; i<wps_length; i++) {
//        wps[i] = i;
//    }
    printf("window=%zu\n", settings.window);
    seq_t d = dtw_warping_paths_full_ndim_twice(wps,  s1,  l1, s2,  l2,
                                                wpsb, s1b, s2b,
                                                true, true, false, 1, &settings);
    printf("d=%f\n", d);
    printf("window=%zu\n", settings.window);
    printf("\n\n");
    dtw_print_wps(wps, l1, l2, &settings);
}

void benchmark15_subsequence2() {
    seq_t s1b[] = {1, 2, 0};  // query
    seq_t s2b[] = {1., 0, 1, 2, 0, 0, 2, 0, 3, 0, 1, 2, 3, 1, 0};
    idx_t l1 = 3;
    idx_t l2 = 15;
    seq_t s1[] = {1., 0.25, -2.};
    seq_t s2[] = {-1., -0.5, 1., 0.25, -1.5, 0.5, 1., -0.75, 1.5, -2., 1., 1., 0.25, -1.75, -1.};
    
    DTWSettings settings = dtw_settings_default();
    settings.window = 0;
    settings.psi_1b = 0;
    settings.psi_1e = 0;
    settings.psi_2b = l2;
    settings.psi_2e = 0;
    settings.penalty = 0.1;
    settings.use_pruning = false;
    idx_t wps_length = dtw_settings_wps_length(l1, l2, &settings);
    printf("wps_length=%zu\n", wps_length);
    seq_t wps[wps_length];
    seq_t wpsb[wps_length];
//    for (idx_t i=0; i<wps_length; i++) {
//        wps[i] = i;
//    }
    printf("window=%zu\n", settings.window);
    seq_t d = dtw_warping_paths_full_ndim_twice(wps,  s1,  l1, s2,  l2,
                                                wpsb, s1b, s2b,
                                                true, true, false, 1, &settings);
    printf("d=%f\n", d);
    printf("window=%zu\n", settings.window);
    printf("\n\n");
    dtw_print_wps(wps, l1, l2, &settings);
    printf("\n\n");
    dtw_print_wps(wpsb, l1, l2, &settings);
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
    
    printf("Negativize r=[4:6], c=[4:5]\n");
    dtw_wps_negativize(&p, wps, l1, l2, 4, 6, 3, 5, true);
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
    dtw_wps_negativize(&p, wps, l1, l2, 1, 7930, 1, 7943, false);
    
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
            print_nb(wps_slice[wpsi]);
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

void benchmark16() {
    double s1[] = {
        -1.7077418 , -1.6659938 , -1.6242079 , -1.6306552 , -1.6816778 ,
        -1.6555744 , -1.6129922 , -1.632736  , -1.7430973 , -1.658726  ,
        -1.6618875 , -1.6533101 , -1.6617085 , -1.6733184 , -1.6762579 ,
        -1.6374508 , -1.6259368 , -1.625577  , -1.6934487 , -1.6595672 ,
        -1.6524769 , -1.6024783 , -1.6642206 , -1.6501966 , -1.6872801 ,
        -1.6427847 , -1.6308933 , -1.6466167 , -1.6584296 , -1.5989099 ,
        -1.6536617 , -1.6646511 , -1.6280719 , -1.6508475 , -1.6697165 ,
        -1.6422188 , -1.6557859 , -1.6933153 , -1.6571033 , -1.6686395 ,
        -1.5931604 , -1.6454781 , -1.7176329 , -1.6133575 , -1.6598295 ,
        -1.6014038 , -1.6372094 , -1.6402063 , -1.6264966 , -1.6827161 ,
        -1.6432177 , -1.5941217 , -1.6427804 , -1.6100383 , -1.6364927 ,
        -1.6543078 , -1.6346395 , -1.629245  , -1.6582006 , -1.623752  ,
        -1.6044234 , -1.5984356 , -1.5703222 , -1.4904046 , -1.470208  ,
        -1.4949865 , -1.4341579 , -1.4613913 , -1.3959086 , -1.3644217 ,
        -1.3122834 , -1.2320485 , -1.1016329 , -1.0390286 , -0.92884027,
        -0.81913935, -0.73657426, -0.61919104, -0.50307291, -0.3170302 ,
        -0.28475764, -0.13154499, -0.04408521,  0.06338575,  0.065106  ,
        0.2088558 ,  0.25554099,  0.31467028,  0.40654056,  0.49369053,
        0.478165  ,  0.54648564,  0.55370479,  0.53808791,  0.64342017,
        0.59228728,  0.62697198,  0.61447923,  0.62333759,  0.65684046,
        0.69147322,  0.62686676,  0.64053882,  0.60856895,  0.65741402,
        0.67571016,  0.66826306,  0.61628674,  0.69409025,  0.64938894,
        0.62712731,  0.70318346,  0.70120732,  0.68662437,  0.6462073 ,
        0.67797827,  0.67059905,  0.62400989,  0.6741306 ,  0.61104008,
        0.690703  ,  0.70885342,  0.66768231,  0.61848321,  0.67230825,
        0.68474772,  0.57783173,  0.62292319,  0.66664016,  0.60273481,
        0.67171029,  0.59588248,  0.65848862,  0.63901874,  0.64952114,
        0.65787405,  0.62005147,  0.66634982,  0.70509728,  0.70641284,
        0.66325597,  0.63261875,  0.61732953,  0.64922335,  0.67322157,
        0.64412546,  0.65521344,  0.65050511,  0.70471415,  0.67540128,
        0.6780457 ,  0.64860001,  0.69141342,  0.64360064,  0.65483869,
        0.69154684,  0.61952224,  0.64579356,  0.69293025,  0.61967024,
        0.63285046,  0.66700962,  0.62629091,  0.66064571,  0.66884969,
        0.63167467,  0.72150801,  0.60772162,  0.63799983,  0.63524151,
        0.65328398,  0.64638452,  0.68359583,  0.74989888,  0.66446511,
        0.65618735,  0.6547654 ,  0.65828109,  0.60721062,  0.7061042 ,
        0.67200664,  0.68283273,  0.72453649,  0.59941649,  0.64682662,
        0.70720243,  0.59104577,  0.60090293,  0.65778189,  0.68036652,
        0.67408786,  0.67503554,  0.65544948,  0.66959667,  0.65638392,
        0.64495299,  0.62708697,  0.64987108,  0.66841922,  0.73429934,
        0.61711176,  0.69283453,  0.69830912,  0.63462652,  0.66195505,
        0.70352493,  0.65674452,  0.63392397,  0.62936258,  0.67540534,
        0.70686178,  0.67467167,  0.70714473,  0.63621398,  0.64621586,
        0.62336068,  0.67603676,  0.62602499,  0.64453625,  0.66646546,
        0.63872028,  0.66845338,  0.71351754,  0.63382118,  0.6417104 ,
        0.67204915,  0.67405262,  0.62638244,  0.63335212,  0.6225208 ,
        0.64718448,  0.67924997,  0.65828833,  0.66751651,  0.64699996,
        0.6350338 ,  0.62708547,  0.64954412,  0.66964353,  0.64337554,
        0.65387452,  0.68448435,  0.64654273,  0.61383799,  0.67124294,
        0.63082385,  0.64692715,  0.66437067,  0.67989974,  0.67205521,
        0.68601371,  0.64755019,  0.62023155,  0.7005264 ,  0.63000998,
        0.66893332,  0.66855475,  0.6618327 ,  0.67937027,  0.62875688,
        0.65838715,  0.67064291,  0.71399787,  0.64397341,  0.69421788,
        0.64638927,  0.64824028,  0.68500174,  0.65417708,  0.66127217,
        0.6486727 ,  0.69447091,  0.64257381,  0.65556164,  0.70985487
    };
    double s2[] = {
        -2.2734101 , -2.3318273 , -2.3096042 , -2.3595022 , -2.3399536 ,
        -2.278084  , -2.297286  , -2.3376906 , -2.3399578 , -2.3323502 ,
        -2.3011074 , -2.3014387 , -2.3573341 , -2.295304  , -2.2671552 ,
        -2.2859158 , -2.3368916 , -2.3525993 , -2.3001195 , -2.3212602 ,
        -2.3199022 , -2.3625251 , -2.2823663 , -2.2700766 , -2.3405575 ,
        -2.2940084 , -2.2981518 , -2.3923028 , -2.2964382 , -2.2230191 ,
        -2.2786015 , -2.2567051 , -2.1655248 , -2.2102227 , -2.1365554 ,
        -2.118938  , -2.0543114 , -1.9985772 , -2.0070635 , -1.9759307 ,
        -1.8231342 , -1.7919128 , -1.6720546 , -1.4263591 , -1.3463522 ,
        -1.185239  , -1.0132148 , -0.89394598, -0.77857472, -0.59279926,
        -0.48093192, -0.39805739, -0.28155469, -0.1513016 , -0.04497553,
        0.03827154,  0.14892885,  0.1671    ,  0.21011189,  0.20880485,
        0.29635416,  0.38716629,  0.36688057,  0.43232579,  0.36953298,
        0.39704181,  0.44019568,  0.47339334,  0.43006408,  0.42098565,
        0.4262782 ,  0.40920715,  0.50854796,  0.50318497,  0.48890449,
        0.42446421,  0.49082081,  0.47501626,  0.42596525,  0.52370387,
        0.54987416,  0.5031069 ,  0.48925523,  0.50435565,  0.49747049,
        0.47645442,  0.50909073,  0.45870796,  0.51274729,  0.47419533,
        0.44190899,  0.58449525,  0.43772841,  0.47949696,  0.46467423,
        0.45880694,  0.50908702,  0.47816314,  0.52612744,  0.53574151,
        0.49864036,  0.49324678,  0.55930111,  0.53030943,  0.50973665,
        0.50234161,  0.44073499,  0.53011945,  0.56523874,  0.50856565,
        0.52085188,  0.49574036,  0.51342444,  0.53139452,  0.53686443,
        0.43894685,  0.52263814,  0.52757224,  0.51023894,  0.47409317,
        0.52181135,  0.45173629,  0.45199642,  0.44980806,  0.47755163,
        0.52578747,  0.49385939,  0.49125525,  0.48620789,  0.48995821,
        0.5328419 ,  0.47156359,  0.49557518,  0.43783697,  0.42865407,
        0.52084425,  0.49193226,  0.48866177,  0.45774266,  0.51246829,
        0.51392861,  0.43896   ,  0.46218793,  0.52278103,  0.46516462,
        0.45794104,  0.43525649,  0.49347604,  0.47856621,  0.47270358,
        0.49527195,  0.54407105,  0.50411459,  0.48873273,  0.48522122,
        0.45120408,  0.48209694,  0.51214892,  0.52945381,  0.44482478,
        0.49064292,  0.44003353,  0.47095229,  0.52719882,  0.49249499,
        0.47299445,  0.5265269 ,  0.43266343,  0.46536841,  0.4890501 ,
        0.48831628,  0.44142751,  0.47269008,  0.48317535,  0.51777539,
        0.4852126 ,  0.43866068,  0.4822576 ,  0.49477842,  0.50804807,
        0.47917401,  0.44049421,  0.52510258,  0.49213577,  0.48479442,
        0.51959567,  0.53418278,  0.53655142,  0.48990709,  0.55301341,
        0.53839897,  0.41781022,  0.51178125,  0.48502954,  0.50328745,
        0.45151215,  0.47566674,  0.42414921,  0.46614317,  0.41315709,
        0.51630431,  0.46229539,  0.47067209,  0.46865451,  0.45301627,
        0.46760726,  0.48863195,  0.51588916,  0.53752908,  0.4758819 ,
        0.50973778,  0.4652192 ,  0.49274421,  0.48136782,  0.44518397,
        0.36383041,  0.51629035,  0.49855855,  0.4957362 ,  0.51923751,
        0.52253142,  0.51171643,  0.50712265,  0.3998718 ,  0.43929912,
        0.4706079 ,  0.4873933 ,  0.43906721,  0.4956588 ,  0.4827489 ,
        0.46636476,  0.49855251,  0.49865191,  0.47906694,  0.46104385,
        0.45298359,  0.48616642,  0.48104124,  0.53370038,  0.48669128,
        0.504324  ,  0.44355437,  0.45969647,  0.50480117,  0.49613569,
        0.49314791,  0.44922423,  0.45696984,  0.50356553,  0.53579012,
        0.4952385 ,  0.45381791,  0.49119533,  0.53561095,  0.50571525,
        0.52827123,  0.51314248,  0.52819397,  0.46780699,  0.45597477,
        0.50776797,  0.49174659,  0.48589964,  0.38088018,  0.47694901,
        0.49763412,  0.50463642,  0.47196648,  0.4400466 ,  0.45180175,
        0.46128678,  0.50638177,  0.4518287 ,  0.55252568,  0.44968947
    };
    idx_t l1 = 275;
    idx_t l2 = 275;
    DTWSettings settings = dtw_settings_default();
    settings.use_pruning = true;
    settings.max_dist = 10.288354487238024;
//    settings.window = 0;
    
    double dd = dtw_distance(s2, l2, s1, l1, &settings);
    printf("dd=%f\n", dd);
    
    idx_t wps_length = dtw_settings_wps_length(l1, l2, &settings);
    printf("wps_length=%zu\n", wps_length);
    seq_t wps[wps_length];
    seq_t d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, true, &settings);
    printf("d=%f\n", d);
//    dtw_print_wps_compact(wps, l1, l2, &settings);
//    printf("\n\n");
//    dtw_print_wps(wps, l1, l2, &settings);
    idx_t i1[l1+l2];
    idx_t i2[l1+l2];
    for (idx_t i=0; i<(l1+l2); i++) {i1[i]=0; i2[i]=0;}
    dtw_best_path(wps, i1, i2, l1, l2, &settings);
    printf("best_path = [");
    for (idx_t i=0; i<(l1+l2); i++) {
        printf("(%zu,%zu)", i1[i], i2[i]);
    }
    printf("]\n");
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
//    benchmark4();
//    benchmark5();
//    benchmark6();
//    benchmark7();
//    benchmark8();
//    benchmark9();
//    benchmark10();
    benchmark11b();
//    benchmark12_subsequence();
//    benchmark13();
//    benchmark14();
//    benchmark15_subsequence();
//    benchmark15_subsequence2();
//    benchmark_loco();
//    benchmark_affinity();
//    wps_test();
//    benchmark16();
    
    time(&end_t);
    clock_gettime(CLOCK_REALTIME, &end);
    diff_t = difftime(end_t, start_t);
    diff_t2 = ((double)end.tv_sec*1e9 + end.tv_nsec) - ((double)start.tv_sec*1e9 + start.tv_nsec);
    printf("Execution time = %f sec / %f ms\n", diff_t, diff_t2/1000000);
    
    return 0;
}


