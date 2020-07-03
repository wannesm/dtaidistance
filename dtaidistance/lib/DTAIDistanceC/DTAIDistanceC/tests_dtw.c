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

#include "dtw.h"


//#define SKIPALL


//----------------------------------------------------
// MARK: DTW

Test(dtw, test_a_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    DTWSettings settings = dtw_settings_default();
    double d = dtw_distance(s1, 9, s2, 9, &settings);
//    printf("d=%f\n", d);
    cr_assert_float_eq(d, sqrt(2), 0.001);
}

Test(dtw, test_b_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.};
    double s2[] = {0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.};
    DTWSettings settings = dtw_settings_default();
    settings.window = 3;
    double d = dtw_distance(s1, 12, s2, 11, &settings);
    cr_assert_float_eq(d, 0.02, 0.001);
    dtw_printprecision_reset();
}

Test(dtw, test_b_b) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.};
    double s2[] = {0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.};
    DTWSettings settings = dtw_settings_default();
    double d = dtw_distance(s1, 12, s2, 11, &settings);
//    printf("d = %f\n", d);
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
}

//----------------------------------------------------
// MARK: WPS

Test(wps, test_a_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    DTWSettings settings = dtw_settings_default();
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 10*10);
    double d = dtw_warping_paths(wps, s1, 9, s2, 9, true, true, &settings);
//    printf("d = %f\n", d);
    free(wps);
    cr_assert_float_eq(d, sqrt(2), 0.001);
}

Test(wps, test_b_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.};
    double s2[] = {0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.};
    DTWSettings settings = dtw_settings_default();
    settings.window = 3;
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 13*12);
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
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 13*12);
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
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 8*8);
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
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 8*8);
    double d = dtw_warping_paths(wps, s1, 7, s2, 7, true, true, &settings);
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
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 8*8);
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
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 8*8);
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
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 10*10);
    double d = dtw_warping_paths(wps, s1, 9, s2, 9, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 3037000496.440516, 0.001);
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
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 41*41);
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
    dtwvalue * wps = (dtwvalue *)malloc(sizeof(dtwvalue) * 41*41);
    double d = dtw_warping_paths(wps, s2, 40, s1, 40, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 0.0, 0.001);
}
