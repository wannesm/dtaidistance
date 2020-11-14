//
//  tests_dtw.c
//  DTAIDistance
//
//  Unit tests, depends on https://criterion.readthedocs.io
//
//  Copyright © 2020 Wannes Meert.
//  Apache License, Version 2.0, see LICENSE for details.
//

#include <stdio.h>
#include <math.h>
#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include "dd_dtw.h"


//#define SKIPALL


//----------------------------------------------------
// MARK: DTW

seq_t dtw_warping_paths_distance(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, DTWSettings *settings) {
    idx_t length = dtw_settings_wps_length(l1, l2, settings);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * length);
    seq_t d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, settings);
    free(wps);
    return d;
}

struct dtw_test_params {
    DTWFnPtr fn;
    DTWSettings settings;
    int id;
};


ParameterizedTestParameters(dtw, test_series1) {
    static struct dtw_test_params params[] = {
        {.fn = dtw_distance, .settings={.window=0}, .id=0},
        {.fn = dtw_warping_paths_distance, .settings={.window=0}, .id=1},
        {.fn = dtw_distance, .settings={.window=0, .use_pruning=true}, .id=2}
    };
    idx_t nb_params = sizeof (params) / sizeof (struct dtw_test_params);
    return cr_make_param_array(struct dtw_test_params, params, nb_params);
}

ParameterizedTest(struct dtw_test_params *param, dtw, test_series1) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    double d = param->fn(s1, 9, s2, 9, &param->settings);
//    printf("d=%f\n", d);
    cr_assert_float_eq(d, sqrt(2), 0.001);
}


ParameterizedTestParameters(dtw, test_series2) {
    static struct dtw_test_params params[] = {
        {.fn = dtw_distance, .settings={.window=0}, .id=0},
        {.fn = dtw_warping_paths_distance, .settings={.window=0}, .id=1},
        {.fn = dtw_distance, .settings={.window=0, .use_pruning=true}, .id=2},
        {.fn = dtw_distance, .settings={.window=3}, .id=3},
        {.fn = dtw_warping_paths_distance, .settings={.window=3}, .id=4},
        {.fn = dtw_distance, .settings={.window=3, .use_pruning=true}, .id=5}
    };
    idx_t nb_params = sizeof (params) / sizeof (struct dtw_test_params);
    return cr_make_param_array(struct dtw_test_params, params, nb_params);
}

ParameterizedTest(struct dtw_test_params *param, dtw, test_series2) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.};
    double s2[] = {0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.};
    double d = param->fn(s1, 12, s2, 11, &param->settings);
    cr_assert_float_eq(d, 0.02, 0.001);
    dtw_printprecision_reset();
}


Test(dtw, test_c_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_dist = 1.1;
    double d = dtw_distance(s1, 7, s2, 7, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);
    d = dtw_warping_paths_distance(s1, 7, s2, 7, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);
}

Test(dtw, test_c_b) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_dist = 0.1;
    double d = dtw_distance(s1, 7, s2, 7, &settings);
    cr_assert(isinf(d));
    d = dtw_warping_paths_distance(s1, 7, s2, 7, &settings);
    cr_assert(isinf(d));
}

Test(dtw, test_c_c) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_step = 1.1;
    double d = dtw_distance(s1, 7, s2, 7, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);
    d = dtw_warping_paths_distance(s1, 7, s2, 7, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);
}

Test(dtw, test_c_d) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_step = 0.1;
    double d = dtw_distance(s1, 7, s2, 7, &settings);
    cr_assert(isinf(d));
    d = dtw_warping_paths_distance(s1, 7, s2, 7, &settings);
    cr_assert(isinf(d));
}

Test(dtw, test_d_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double maxval_thirtytwobit = 2147483647;
    double s1[] = {maxval_thirtytwobit, maxval_thirtytwobit, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {1., 2, 0, 0, 0, 0, 0, 1, 0};
    DTWSettings settings = dtw_settings_default();
    double d = dtw_distance(s1, 9, s2, 9, &settings);
    cr_assert_float_eq(d, 3037000496.440516, 0.001);
    d = dtw_warping_paths_distance(s1, 9, s2, 9, &settings);
    cr_assert_float_eq(d, 3037000496.440516, 0.001);
}

// MARK DTW - PrunedDTW

Test(dtwp, test_c_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.use_pruning = true;
    double d = dtw_distance(s1, 7, s2, 7, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);
    d = dtw_warping_paths_distance(s1, 7, s2, 7, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);
}

Test(dtwp, test_c_c) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_step = 1.1;
    settings.use_pruning = true;
    double d = dtw_distance(s1, 7, s2, 7, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);
    d = dtw_warping_paths_distance(s1, 7, s2, 7, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);
}

Test(dtwp, test_c_d) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_step = 0.1;
    settings.use_pruning = true;
    double d = dtw_distance(s1, 7, s2, 7, &settings);
    cr_assert(isinf(d));
    d = dtw_warping_paths_distance(s1, 7, s2, 7, &settings);
    cr_assert(isinf(d));
}

Test(dtwp, test_d_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double maxval_thirtytwobit = 2147483647;
    double s1[] = {maxval_thirtytwobit, maxval_thirtytwobit, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {1., 2, 0, 0, 0, 0, 0, 1, 0};
    DTWSettings settings = dtw_settings_default();
    settings.use_pruning = true;
    double d = dtw_distance(s1, 9, s2, 9, &settings);
    cr_assert_float_eq(d, 3037000496.440516, 0.001);
    d = dtw_warping_paths_distance(s1, 9, s2, 9, &settings);
    cr_assert_float_eq(d, 3037000496.440516, 0.001);
}

// MARK: DTW - PSI

Test(dtw_psi, test_a_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,-0.28,0.22,
                   0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,-0.07,0.42,
                   0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34,0.15,0.61};
    double s2[] = {-0.84,-0.48,0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,
                   -0.28,0.22,0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,
                  -0.07,0.42,0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34};
    DTWSettings settings = dtw_settings_default();
    settings.psi = 2;
    double d = dtw_distance(s1, 40, s2, 40, &settings);
    cr_assert_float_eq(d, 0.0, 0.001);
    d = dtw_warping_paths_distance(s1, 40, s2, 40, &settings);
    cr_assert_float_eq(d, 0.0, 0.001);
}

Test(dtw_psi, test_a_b) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,-0.28,0.22,
                   0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,-0.07,0.42,
                   0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34,0.15,0.61};
    double s2[] = {-0.84,-0.48,0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,
                   -0.28,0.22,0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,
                  -0.07,0.42,0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34};
    DTWSettings settings = dtw_settings_default();
    settings.psi = 2;
    double d = dtw_distance(s2, 40, s1, 40, &settings);
    cr_assert_float_eq(d, 0.0, 0.001);
    d = dtw_warping_paths_distance(s2, 40, s1, 40, &settings);
    cr_assert_float_eq(d, 0.0, 0.001);
}

//----------------------------------------------------
// MARK: WPS

Test(wps, test_b_a) {
    #ifdef SKIPALL
//    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.};
    double s2[] = {0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.};
    DTWSettings settings = dtw_settings_default();
    settings.window = 3;
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 13*12);
    double d = dtw_warping_paths(wps, s1, 12, s2, 11, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 0.02, 0.001);
    dtw_printprecision_reset();
}

Test(wps, test_b_b) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.};
    double s2[] = {0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.};
    DTWSettings settings = dtw_settings_default();
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 13*12);
    double d = dtw_warping_paths(wps, s1, 12, s2, 11, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 0.02, 0.001);
    dtw_printprecision_reset();
}

Test(wps, test_c_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_dist = 1.1;
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 8*8);
    double d = dtw_warping_paths(wps, s1, 7, s2, 7, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 1.0, 0.001);
}

Test(wps, test_c_b) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_dist = 0.1;
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 8*8);
    double d = dtw_warping_paths(wps, s1, 7, s2, 7, true, true, &settings);
//    dtw_print_wps(wps, 7, 7);
    free(wps);
    cr_assert(isinf(d));
}

Test(wps, test_c_c) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_step = 1.1;
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 8*8);
    double d = dtw_warping_paths(wps, s1, 7, s2, 7, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 1.0, 0.001);
}

Test(wps, test_c_d) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0};
    double s2[] = {0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    DTWSettings settings = dtw_settings_default();
    settings.max_step = 0.1;
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 8*8);
    double d = dtw_warping_paths(wps, s1, 7, s2, 7, true, true, &settings);
    free(wps);
    cr_assert(isinf(d));
}

Test(wps, test_d_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double maxval_thirtytwobit = 2147483647;
    double s1[] = {maxval_thirtytwobit, maxval_thirtytwobit, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {1., 2, 0, 0, 0, 0, 0, 1, 0};
    DTWSettings settings = dtw_settings_default();
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 10*10);
    double d = dtw_warping_paths(wps, s1, 9, s2, 9, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 3037000496.440516, 0.001);
}

Test(wps, test_e_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    seq_t s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0};
    seq_t s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0};
    idx_t l1 = 11;
    idx_t l2 = 11;

    DTWSettings settings = dtw_settings_default();
    idx_t i1[l1+l2];
    idx_t i2[l1+l2];
    for (idx_t i=0; i<l1+l2; i++) {
        i1[i] = 0;
        i2[i] = 0;
    }
    warping_path(s1, l1, s2, l2, i1, i2, &settings);
    
    idx_t r1[] = {10,9,8,7,6,5,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0};
    idx_t r2[] = {10,9,8,7,6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0};
    
//    for (idx_t i=0; i<l1+l2; i++) {
//        printf("%zu %zu\n", i1[i], r1[i]);
//        cr_assert_eq(i1[i], r1[i]);
//        cr_assert_eq(i2[i], r2[i]);
//    }

    cr_assert_arr_eq(i1, r1, sizeof(r1));
    cr_assert_arr_eq(i2, r2, sizeof(r2));
}


//----------------------------------------------------
// MARK: WPS - PSI

Test(wps_psi, test_a_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,-0.28,0.22,
                   0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,-0.07,0.42,
                   0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34,0.15,0.61};
    double s2[] = {-0.84,-0.48,0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,
                   -0.28,0.22,0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,
                  -0.07,0.42,0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34};
    DTWSettings settings = dtw_settings_default();
    settings.psi = 2;
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 41*41);
    double d = dtw_warping_paths(wps, s1, 40, s2, 40, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 0.0, 0.001);
}

Test(wps_psi, test_a_b) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,-0.28,0.22,
                   0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,-0.07,0.42,
                   0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34,0.15,0.61};
    double s2[] = {-0.84,-0.48,0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,
                   -0.28,0.22,0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,
                  -0.07,0.42,0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34};
    DTWSettings settings = dtw_settings_default();
    settings.psi = 2;
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 41*41);
    double d = dtw_warping_paths(wps, s2, 40, s1, 40, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 0.0, 0.001);
}

// MARK: NDIM

Test(ndim, test_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0, 0, 0, 1, 2, 1, 0,  1, 0, 0};
    double s2[] = {0, 0, 2, 1, 0, 1, 0, .5, 0, 0};
    DTWSettings settings = dtw_settings_default();
    seq_t d = dtw_distance_ndim(s1, 5, s2, 5, 2, &settings);
//    printf("d=%f\n", d);
    cr_assert_float_eq(d, 1.118033988749895, 0.001);
}

//----------------------------------------------------
// MARK: DBA

Test(dba, test_a_matrix) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    
    double s[] = {
        0.5, 1, 2, 3, 2.0, 2.1, 1.0, 0, 0, 0, // Row 0
        0.4, 0, 1, 1.5, 1.9, 2.0, 0.9, 1, 0, 0 // Row 1
    };
    double exp_avg[] = {0.25, 1.1666666666666667, 1.95, 2.5, 2.0, 2.05, 0.9666666666666667, 0.0, 0.0, 0.0};
    idx_t nb_cols = 10;
    idx_t nb_rows = 2;
    seq_t c[nb_cols];
    for (idx_t i=0; i<nb_cols; i++) { // Copy first series
        c[i] = s[i];
    }
//    bit_array(mask, nb_rows)
    ba_t mask[bit_bytes(nb_rows)];
    for (int i=0; i<nb_rows; i++) {mask[i]=0;}
    bit_set(mask, 0);
    bit_set(mask, 1);
    DTWSettings settings = dtw_settings_default();
    
    dtw_dba_matrix(s, nb_rows, nb_cols, c, nb_cols, mask, &settings);
    
    for (idx_t i=0; i<nb_cols; i++) {
        cr_assert_float_eq(c[i], exp_avg[i], 0.001);
    }
}

Test(dba, test_a_ptrs) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    
    double s1[] = {0.5, 1, 2, 3, 2.0, 2.1, 1.0, 0, 0, 0};
    double s2[] = {0.4, 0, 1, 1.5, 1.9, 2.0, 0.9, 1, 0, 0};
    double **s = (double **)malloc(2 * sizeof(double *));
    s[0] = s1;
    s[1] = s2;
    
    double exp_avg[] = {0.25, 1.1666666666666667, 1.95, 2.5, 2.0, 2.05, 0.9666666666666667, 0.0, 0.0, 0.0};
    idx_t nb_cols = 10;
    idx_t nb_rows = 2;
    idx_t lengths[2] = {nb_cols, nb_cols};
    seq_t c[nb_cols];
    for (idx_t i=0; i<nb_cols; i++) { // Copy first series
        c[i] = s[0][i];
    }
//    bit_array(mask, nb_rows)
    ba_t mask[bit_bytes(nb_rows)];
    for (int i=0; i<nb_rows; i++) {mask[i]=0;}
    bit_set(mask, 0);
    bit_set(mask, 1);
    DTWSettings settings = dtw_settings_default();
        
    dtw_dba_ptrs(s, nb_rows, lengths, c, nb_cols, mask, &settings);
    
    for (idx_t i=0; i<nb_cols; i++) {
        cr_assert_float_eq(c[i], exp_avg[i], 0.001);
    }
    
    free(s);
}

