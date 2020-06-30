//
//  tests_matrix.c
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
#include "dtw_openmp.h"


//#define SKIPALL

//----------------------------------------------------
// MARK: MATRIX

Test(matrix, test_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    double *s[] = {s1, s2};
    int lengths[] = {9, 9};
    double result[1];
    DTWSettings settings = dtw_default_settings();
    DTWBlock block = dtw_empty_block();
    dtw_distances_ptrs(s, 2, lengths, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
}

Test(matrix, test_a_parallel) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    double *s[] = {s1, s2};
    int lengths[] = {9, 9};
    double result[1];
    DTWSettings settings = dtw_default_settings();
    DTWBlock block = dtw_empty_block();
    dtw_distances_ptrs_parallel(s, 2, lengths, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
}

Test(matrix, test_b) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    double s3[] = {1., 2, 0, 0, 0, 0, 0, 1};
    double *s[] = {s1, s2, s3};
    int lengths[] = {9, 9, 8};
    double result[3];
    DTWSettings settings = dtw_default_settings();
    DTWBlock block = dtw_empty_block();
    dtw_distances_ptrs(s, 3, lengths, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
}

Test(matrix, test_b_parallel) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    double s3[] = {1., 2, 0, 0, 0, 0, 0, 1};
    double *s[] = {s1, s2, s3};
    int lengths[] = {9, 9, 8};
    double result[3];
    DTWSettings settings = dtw_default_settings();
    DTWBlock block = dtw_empty_block();
    dtw_distances_ptrs_parallel(s, 3, lengths, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
}

Test(matrix, test_c_block_ptrs) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0., 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0., 1, 2, 0, 0, 0, 0, 0, 0};
    double s3[] = {1., 2, 0, 0, 0, 0, 0, 1, 1};
    double s4[] = {0., 0, 1, 2, 1, 0, 1, 0, 0};
    double s5[] = {0., 1, 2, 0, 0, 0, 0, 0, 0};
    double s6[] = {1., 2, 0, 0, 0, 0, 0, 1, 1};
    double *s[] = {s1, s2, s3, s4, s5, s6};
    int lengths[] = {9, 9, 9, 9, 9, 9};
    double result[5];
    DTWSettings settings = dtw_default_settings();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5};
    
    dtw_distances_ptrs(s, 6, lengths, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
    cr_assert_float_eq(result[1], 0, 0.001);
    cr_assert_float_eq(result[2], 2.23606798, 0.001);
    cr_assert_float_eq(result[3], 1.73205081, 0.001);
    cr_assert_float_eq(result[4], sqrt(2), 0.001);
}

Test(matrix, test_c_block_matrix) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s[] = {0., 0, 1, 2, 1, 0, 1, 0, 0,
                  0., 1, 2, 0, 0, 0, 0, 0, 0,
                  1., 2, 0, 0, 0, 0, 0, 1, 1,
                  0., 0, 1, 2, 1, 0, 1, 0, 0,
                  0., 1, 2, 0, 0, 0, 0, 0, 0,
                  1., 2, 0, 0, 0, 0, 0, 1, 1};
    int nb_cols = 9;
    int nb_rows = 6;
    double result[5];
    DTWSettings settings = dtw_default_settings();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5};
    dtw_distances_matrix(s, nb_rows, nb_cols, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
    cr_assert_float_eq(result[1], 0, 0.001);
    cr_assert_float_eq(result[2], 2.23606798, 0.001);
    cr_assert_float_eq(result[3], 1.73205081, 0.001);
    cr_assert_float_eq(result[4], sqrt(2), 0.001);
}

Test(matrix, test_c_block_ptrs_parallel) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0., 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0., 1, 2, 0, 0, 0, 0, 0, 0};
    double s3[] = {1., 2, 0, 0, 0, 0, 0, 1, 1};
    double s4[] = {0., 0, 1, 2, 1, 0, 1, 0, 0};
    double s5[] = {0., 1, 2, 0, 0, 0, 0, 0, 0};
    double s6[] = {1., 2, 0, 0, 0, 0, 0, 1, 1};
    double *s[] = {s1, s2, s3, s4, s5, s6};
    int lengths[] = {9, 9, 9, 9, 9, 9};
    double result[5];
    DTWSettings settings = dtw_default_settings();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5};
    for (int j=0; j<1000; j++) {
//        printf("---START---\n");
        dtw_distances_ptrs_parallel(s, 6, lengths, result, &block, &settings);
//        for (int i=0; i<5; i++) {
//            printf("%.4f ", result[i]);
//        }
//        printf("\n");
        cr_assert_float_eq(result[0], 1.41421356, 0.001);
        cr_assert_float_eq(result[1], 0.00000000, 0.001);
        cr_assert_float_eq(result[2], 2.23606798, 0.001);
        cr_assert_float_eq(result[3], 1.73205081, 0.001);
        cr_assert_float_eq(result[4], 1.41421356, 0.001);
    }
}

Test(matrix, test_c_block_matrix_parallel) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s[] = {0., 0, 1, 2, 1, 0, 1, 0, 0,
                  0., 1, 2, 0, 0, 0, 0, 0, 0,
                  1., 2, 0, 0, 0, 0, 0, 1, 1,
                  0., 0, 1, 2, 1, 0, 1, 0, 0,
                  0., 1, 2, 0, 0, 0, 0, 0, 0,
                  1., 2, 0, 0, 0, 0, 0, 1, 1};
    int nb_cols = 9;
    int nb_rows = 6;
    double result[5];
    DTWSettings settings = dtw_default_settings();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5};
    dtw_distances_matrix_parallel(s, nb_rows, nb_cols, result, &block, &settings);
//    for (int i=0; i<5; i++) {
//        printf("%f ", result[i]);
//    }
//    printf("\n");
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
    cr_assert_float_eq(result[1], 0, 0.001);
    cr_assert_float_eq(result[2], 2.23606798, 0.001);
    cr_assert_float_eq(result[3], 1.73205081, 0.001);
    cr_assert_float_eq(result[4], sqrt(2), 0.001);
}
