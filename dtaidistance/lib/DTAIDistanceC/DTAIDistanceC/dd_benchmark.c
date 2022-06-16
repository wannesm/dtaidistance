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

void benchmark1() {
    int size=10000;
   double ra1[size], ra2[size];
   int i;
   for (i=0; i<size; i++) {
       ra1[i] = rand() % 10;
       ra2[i] = rand() % 10;
   }
   
   DTWSettings settings = dtw_settings_default();
   double d = dtw_distance(ra1, size, ra2, size, &settings);
   
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
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    DTWSettings settings = dtw_settings_default();
    settings.use_pruning = true;
    double d = dtw_distance(s1, 9, s2, 9, &settings);
    printf("d=%f\n", d);
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
    double result[dtw_distances_length(&block, 6)];
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
    ssize_t from_l = 276;
    double from_s[] = {1.000000, 0.544070, 0.657863, 0.611239, 0.545413, 0.532156, 0.593013, 0.595280, 0.538277, 0.563429, 0.595956, 0.632915, 0.584646, 0.623187, 0.605955, 0.590372, 0.627021, 0.613486, 0.623053, 0.546018, 0.641136, 0.549395, 0.610697, 0.577974, 0.635666, 0.574218, 0.606361, 0.638053, 0.640597, 0.605811, 0.654865, 0.629337, 0.673546, 0.640830, 0.671068, 0.595392, 0.589426, 0.615870, 0.667038, 0.603801, 0.633861, 0.647193, 0.644830, 0.643686, 0.623500, 0.562371, 0.610632, 0.607732, 0.642409, 0.577916, 0.686207, 0.612979, 1.008782, 2.308254, 3.486516, 3.827367, 3.817775, 1.011998, -1.248066, -1.879353, -1.948347, -1.901329, -1.919121, -1.922147, -1.981481, -1.971905, -1.987526, -1.879364, -1.900581, -1.882505, -1.873600, -1.887615, -1.950244, -1.890422, -1.878405, -1.860656, -1.909900, -1.866172, -1.881603, -1.831401, -1.782852, -1.761904, -1.843545, -1.821094, -1.793479, -1.738491, -1.714164, -1.781086, -1.701718, -1.758649, -1.636158, -1.676155, -1.635285, -1.664891, -1.653706, -1.599270, -1.585701, -1.570689, -1.556533, -1.525705, -1.534395, -1.514424, -1.458524, -1.434051, -1.399409, -1.439524, -1.388384, -1.314108, -1.287314, -1.243788, -1.266220, -1.217401, -1.140410, -1.077915, -1.036099, -1.023596, -0.941927, -0.964608, -0.926929, -0.806925, -0.815029, -0.802277, -0.777635, -0.716090, -0.718716, -0.598533, -0.622565, -0.642872, -0.617758, -0.554491, -0.510171, -0.502370, -0.387559, -0.408328, -0.360836, -0.336752, -0.271082, -0.231045, -0.255034, -0.195860, -0.131653, -0.091147, -0.019169, -0.133735, -0.076408, -0.063702, -0.058617, -0.010619, 0.071344, 0.021106, 0.102681, 0.119364, 0.118388, 0.150217, 0.146642, 0.149047, 0.161994, 0.241047, 0.237758, 0.275506, 0.318240, 0.378176, 0.376945, 0.304307, 0.358492, 0.292572, 0.337857, 0.372190, 0.370101, 0.404809, 0.370491, 0.415970, 0.403825, 0.436792, 0.406125, 0.401529, 0.476474, 0.441693, 0.443843, 0.385452, 0.463724, 0.446300, 0.444850, 0.483742, 0.485856, 0.534248, 0.538898, 0.464345, 0.513746, 0.494470, 0.539834, 0.522132, 0.518813, 0.524532, 0.563958, 0.553137, 0.601160, 0.551513, 0.547417, 0.546696, 0.586591, 0.526562, 0.545691, 0.572212, 0.520881, 0.581978, 0.570224, 0.566690, 0.570693, 0.569991, 0.545721, 0.597788, 0.572688, 0.571803, 0.635557, 0.500596, 0.551952, 0.591300, 0.551031, 0.572876, 0.558675, 0.592231, 0.584321, 0.570917, 0.605592, 0.656936, 0.565111, 0.552405, 0.564911, 0.583119, 0.633982, 0.579977, 0.625352, 0.589820, 0.575724, 0.605493, 0.535555, 0.549415, 0.553527, 0.595383, 0.534560, 0.530545, 0.611951, 0.586050, 0.545310, 0.550911, 0.563941, 0.609545, 0.536540, 0.590553, 0.553134, 0.588189, 0.533342, 0.587944, 0.582966, 0.598148, 0.600321, 0.566576, 0.671940, 0.621281, 0.555147, 0.581241, 0.674639, 0.576070, 0.637210, 0.578224, 0.588226, 0.577886, 0.598069, 0.583695, 0.602877, 0.554263, 0.514147, 0.603773, 0.596331, 0.583224};
    ssize_t to_l = 276;
    double to_s[] = {1.000000, 0.544070, 0.657863, 0.611239, 0.545413, 0.532156, 0.593013, 0.595280, 0.538277, 0.563429, 0.595956, 0.632915, 0.584646, 0.623187, 0.605955, 0.590372, 0.627021, 0.613486, 0.623053, 0.546018, 0.641136, 0.549395, 0.610697, 0.577974, 0.635666, 0.574218, 0.606361, 0.638053, 0.640597, 0.605811, 0.654865, 0.629337, 0.673546, 0.640830, 0.671068, 0.595392, 0.589426, 0.615870, 0.667038, 0.603801, 0.633861, 0.647193, 0.644830, 0.643686, 0.623500, 0.562371, 0.610632, 0.607732, 0.642409, 0.577916, 0.686207, 0.612979, 1.008782, 2.308254, 3.486516, 3.827367, 3.817775, 1.011998, -1.248066, -1.879353, -1.948347, -1.901329, -1.919121, -1.922147, -1.981481, -1.971905, -1.987526, -1.879364, -1.900581, -1.882505, -1.873600, -1.887615, -1.950244, -1.890422, -1.878405, -1.860656, -1.909900, -1.866172, -1.881603, -1.831401, -1.782852, -1.761904, -1.843545, -1.821094, -1.793479, -1.738491, -1.714164, -1.781086, -1.701718, -1.758649, -1.636158, -1.676155, -1.635285, -1.664891, -1.653706, -1.599270, -1.585701, -1.570689, -1.556533, -1.525705, -1.534395, -1.514424, -1.458524, -1.434051, -1.399409, -1.439524, -1.388384, -1.314108, -1.287314, -1.243788, -1.266220, -1.217401, -1.140410, -1.077915, -1.036099, -1.023596, -0.941927, -0.964608, -0.926929, -0.806925, -0.815029, -0.802277, -0.777635, -0.716090, -0.718716, -0.598533, -0.622565, -0.642872, -0.617758, -0.554491, -0.510171, -0.502370, -0.387559, -0.408328, -0.360836, -0.336752, -0.271082, -0.231045, -0.255034, -0.195860, -0.131653, -0.091147, -0.019169, -0.133735, -0.076408, -0.063702, -0.058617, -0.010619, 0.071344, 0.021106, 0.102681, 0.119364, 0.118388, 0.150217, 0.146642, 0.149047, 0.161994, 0.241047, 0.237758, 0.275506, 0.318240, 0.378176, 0.376945, 0.304307, 0.358492, 0.292572, 0.337857, 0.372190, 0.370101, 0.404809, 0.370491, 0.415970, 0.403825, 0.436792, 0.406125, 0.401529, 0.476474, 0.441693, 0.443843, 0.385452, 0.463724, 0.446300, 0.444850, 0.483742, 0.485856, 0.534248, 0.538898, 0.464345, 0.513746, 0.494470, 0.539834, 0.522132, 0.518813, 0.524532, 0.563958, 0.553137, 0.601160, 0.551513, 0.547417, 0.546696, 0.586591, 0.526562, 0.545691, 0.572212, 0.520881, 0.581978, 0.570224, 0.566690, 0.570693, 0.569991, 0.545721, 0.597788, 0.572688, 0.571803, 0.635557, 0.500596, 0.551952, 0.591300, 0.551031, 0.572876, 0.558675, 0.592231, 0.584321, 0.570917, 0.605592, 0.656936, 0.565111, 0.552405, 0.564911, 0.583119, 0.633982, 0.579977, 0.625352, 0.589820, 0.575724, 0.605493, 0.535555, 0.549415, 0.553527, 0.595383, 0.534560, 0.530545, 0.611951, 0.586050, 0.545310, 0.550911, 0.563941, 0.609545, 0.536540, 0.590553, 0.553134, 0.588189, 0.533342, 0.587944, 0.582966, 0.598148, 0.600321, 0.566576, 0.671940, 0.621281, 0.555147, 0.581241, 0.674639, 0.576070, 0.637210, 0.578224, 0.588226, 0.577886, 0.598069, 0.583695, 0.602877, 0.554263, 0.514147, 0.603773, 0.596331, 0.583224};
//    DTWSettings {
//      window = 0
//      max_dist = 0.000000
//      max_step = 0.000000
//      max_length_diff = 0
//      penalty = 0.000000
//      psi = 0
//      use_pruning = 0
//      only_ub = 0
//    }
    idx_t *from_i = (idx_t *)malloc((from_l + to_l) * sizeof(idx_t));
    idx_t *to_i = (idx_t *)malloc((from_l + to_l) * sizeof(idx_t));
    DTWSettings settings = dtw_settings_default();
    warping_path(from_s, from_l, to_s, to_l, from_i, to_i, &settings);
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
    benchmark9();
//    benchmark10();
//    benchmark11();
//    benchmark12_subsequence();
//    benchmark13();
    
    time(&end_t);
    clock_gettime(CLOCK_REALTIME, &end);
    diff_t = difftime(end_t, start_t);
    diff_t2 = ((double)end.tv_sec*1e9 + end.tv_nsec) - ((double)start.tv_sec*1e9 + start.tv_nsec);
    printf("Execution time = %f sec / %f ms\n", diff_t, diff_t2/1000000);
    
    return 0;
}


