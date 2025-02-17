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
#include <stddef.h>
#include <criterion/criterion.h>

#include "dd_dtw.h"
#include "dd_dtw_openmp.h"


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
    idx_t lengths[] = {9, 9};
    double result[1];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = dtw_block_empty();
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
    idx_t lengths[] = {9, 9};
    double result[1];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = dtw_block_empty();
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
    idx_t lengths[] = {9, 9, 8};
    double result[3];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = dtw_block_empty();
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
    idx_t lengths[] = {9, 9, 8};
    double result[3];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = dtw_block_empty();
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
    idx_t lengths[] = {9, 9, 9, 9, 9, 9};
    double result[5];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5, .triu=true};
    
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
    idx_t nb_cols = 9;
    idx_t nb_rows = 6;
    double result[5];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5, .triu=true};
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
    idx_t lengths[] = {9, 9, 9, 9, 9, 9};
    double result[5];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5, .triu=true};
    for (int j=0; j<100; j++) {
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
    idx_t nb_cols = 9;
    idx_t nb_rows = 6;
    double result[5];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5, .triu=true};
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

Test(matrix, test_c_block_triu_matrix_parallel) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s[] = {0., 0, 1, 2, 1, 0, 1, 0, 0,
                  0., 1, 2, 0, 0, 0, 0, 0, 0,
                  1., 2, 0, 0, 0, 0, 0, 1, 1,
                  0., 0, 1, 2, 1, 0, 1, 0, 0,
                  0., 1, 2, 0, 0, 0, 0, 0, 0,
                  1., 2, 0, 0, 0, 0, 0, 1, 1};
    idx_t nb_cols = 9;
    idx_t nb_rows = 6;
    double result[6];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5, .triu=false};
    dtw_distances_matrix_parallel(s, nb_rows, nb_cols, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
    cr_assert_float_eq(result[1], 0, 0.001);
    cr_assert_float_eq(result[2], 2.23606798, 0.001);
    cr_assert_float_eq(result[3], 1.73205081, 0.001);
    cr_assert_float_eq(result[4], 0.0, 0.001);
    cr_assert_float_eq(result[5], sqrt(2), 0.001);
}

Test(matrix, test_c_block_triu_matrices_parallel) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s[] = {0., 0, 1, 2, 1, 0, 1, 0, 0,
                  0., 1, 2, 0, 0, 0, 0, 0, 0,
                  1., 2, 0, 0, 0, 0, 0, 1, 1,
                  0., 0, 1, 2, 1, 0, 1, 0, 0,
                  0., 1, 2, 0, 0, 0, 0, 0, 0,
                  1., 2, 0, 0, 0, 0, 0, 1, 1};
    idx_t nb_cols = 9;
    idx_t nb_rows = 6;
    double result[6];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5, .triu=false};
    dtw_distances_matrices_parallel(s, nb_rows, nb_cols, s, nb_rows, nb_cols, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
    cr_assert_float_eq(result[1], 0, 0.001);
    cr_assert_float_eq(result[2], 2.23606798, 0.001);
    cr_assert_float_eq(result[3], 1.73205081, 0.001);
    cr_assert_float_eq(result[4], 0.0, 0.001);
    cr_assert_float_eq(result[5], sqrt(2), 0.001);
}

Test(matrix, test_d_matrices) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double sr[] = {0., 1, 2, 0, 0, 0, 0, 0, 0,
                   1., 2, 0, 0, 0, 0, 0, 1, 1,
                   0., 0, 1, 2, 1, 0, 1, 0, 0};
    double sc[] = {0., 0, 1, 2, 1, 0, 1, 0, 0,
                   0., 1, 2, 0, 0, 0, 0, 0, 0};
    idx_t nb_cols = 9;
    idx_t nb_rows_r = 3;
    idx_t nb_rows_c = 2;
    double result[6];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = {.rb=0, .re=0, .cb=0, .ce=0, .triu=false};
    dtw_distances_matrices(sr, nb_rows_r, nb_cols, sc, nb_rows_c, nb_cols, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
    cr_assert_float_eq(result[1], 0, 0.001);
    cr_assert_float_eq(result[2], 2.23606798, 0.001);
    cr_assert_float_eq(result[3], 1.73205081, 0.001);
    cr_assert_float_eq(result[4], 0.0, 0.001);
    cr_assert_float_eq(result[5], sqrt(2), 0.001);
}

Test(matrix, test_d_matrices_parallel) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double sr[] = {0., 1, 2, 0, 0, 0, 0, 0, 0,
                   1., 2, 0, 0, 0, 0, 0, 1, 1,
                   0., 0, 1, 2, 1, 0, 1, 0, 0};
    double sc[] = {0., 0, 1, 2, 1, 0, 1, 0, 0,
                   0., 1, 2, 0, 0, 0, 0, 0, 0};
    idx_t nb_cols = 9;
    idx_t nb_rows_r = 3;
    idx_t nb_rows_c = 2;
    double result[6];
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = {.rb=0, .re=0, .cb=0, .ce=0, .triu=false};
    dtw_distances_matrices_parallel(sr, nb_rows_r, nb_cols, sc, nb_rows_c, nb_cols, result, &block, &settings);
    cr_assert_float_eq(result[0], sqrt(2), 0.001);
    cr_assert_float_eq(result[1], 0, 0.001);
    cr_assert_float_eq(result[2], 2.23606798, 0.001);
    cr_assert_float_eq(result[3], 1.73205081, 0.001);
    cr_assert_float_eq(result[4], 0.0, 0.001);
    cr_assert_float_eq(result[5], sqrt(2), 0.001);
}



//----------------------------------------------------
// MARK: NDIM

Test(matrix_ndim, test_a_ptrs) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0., 0, 1, 2, 1, 0, 1, 0};
    double s2[] = {0., 1, 2, 0, 0, 0, 0, 0};
    double s3[] = {1., 2, 0, 0, 0, 0, 0, 1};
    double s4[] = {0., 0, 1, 2, 1, 0, 1, 0};
    double s5[] = {0., 1, 2, 0, 0, 0, 0, 0};
    double s6[] = {1., 2, 0, 0, 0, 0, 0, 1};
    double *s[] = {s1, s2, s3, s4, s5, s6};
    idx_t lengths[] = {4, 4, 4, 4, 4, 4};
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = dtw_block_empty();
    double result[dtw_distances_length(&block, 6, 6)];
    dtw_distances_ndim_ptrs(s, 6, lengths, 2, result, &block, &settings);
//    for (int i=0; i<5; i++) {
//        printf("%.4f ", result[i]);
//    }
//    printf("\n");
    cr_assert_float_eq(result[0], 2.4495, 0.001);
    cr_assert_float_eq(result[1], 3.0000, 0.001);
    cr_assert_float_eq(result[2], 0.0000, 0.001);
    cr_assert_float_eq(result[3], 2.4495, 0.001);
    cr_assert_float_eq(result[4], 3.0000, 0.001);
}


Test(matrix_ndim, test_a_ptrs_parallel) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0., 0, 1, 2, 1, 0, 1, 0};
    double s2[] = {0., 1, 2, 0, 0, 0, 0, 0};
    double s3[] = {1., 2, 0, 0, 0, 0, 0, 1};
    double s4[] = {0., 0, 1, 2, 1, 0, 1, 0};
    double s5[] = {0., 1, 2, 0, 0, 0, 0, 0};
    double s6[] = {1., 2, 0, 0, 0, 0, 0, 1};
    double *s[] = {s1, s2, s3, s4, s5, s6};
    idx_t lengths[] = {4, 4, 4, 4, 4, 4};
    DTWSettings settings = dtw_settings_default();
    DTWBlock block = dtw_block_empty();
    double result[dtw_distances_length(&block, 6, 6)];
    dtw_distances_ndim_ptrs_parallel(s, 6, lengths, 2, result, &block, &settings);
//    for (int i=0; i<5; i++) {
//        printf("%.4f ", result[i]);
//    }
//    printf("\n");
    cr_assert_float_eq(result[0], 2.4495, 0.001);
    cr_assert_float_eq(result[1], 3.0000, 0.001);
    cr_assert_float_eq(result[2], 0.0000, 0.001);
    cr_assert_float_eq(result[3], 2.4495, 0.001);
    cr_assert_float_eq(result[4], 3.0000, 0.001);
}


Test(matrix_ndim, test_a_matrix) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
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
    cr_assert_float_eq(result[0], 2.4495, 0.001);
    cr_assert_float_eq(result[1], 3.0000, 0.001);
    cr_assert_float_eq(result[2], 0.0000, 0.001);
    cr_assert_float_eq(result[3], 2.4495, 0.001);
    cr_assert_float_eq(result[4], 3.0000, 0.001);
}


Test(matrix_ndim, test_a_matrix_parallel) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
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
    dtw_distances_ndim_matrix_parallel(s, 6, 4, 2, result, &block, &settings);
    cr_assert_float_eq(result[0], 2.4495, 0.001);
    cr_assert_float_eq(result[1], 3.0000, 0.001);
    cr_assert_float_eq(result[2], 0.0000, 0.001);
    cr_assert_float_eq(result[3], 2.4495, 0.001);
    cr_assert_float_eq(result[4], 3.0000, 0.001);
}


//----------------------------------------------------
// MARK: DBA

Test(dba_ndim, test_a_matrix) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
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
    ba_t mask[bit_bytes(nb_rows)];
    for (int i=0; i<nb_rows; i++) {
        bit_set(mask, i);
    }
    DTWSettings settings = dtw_settings_default();
    dtw_dba_matrix(s, nb_rows, nb_cols, c, nb_cols, mask, 0, ndim, &settings);
    for (int i=0; i<(nb_cols*ndim); i++) {
        cr_assert_float_eq(c[i], exp_avg[i], 0.001);
    }
}

Test(dba_ndim, test_a_ptrs) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0., 0, 1, 2, 1, 0, 1, 0};
    double s2[] = {0., 1, 2, 0, 0, 0, 0, 0};
    double s3[] = {1., 2, 0, 0, 0, 0, 0, 1};
    double s4[] = {0., 0, 1, 2, 1, 0, 1, 0};
    double s5[] = {0., 1, 2, 0, 0, 0, 0, 0};
    double s6[] = {1., 2, 0, 0, 0, 0, 0, 1};
    double *s[] = {s1, s2, s3, s4, s5, s6};
    idx_t lengths[] = {4, 4, 4, 4, 4, 4};
    double exp_avg[] = {0.33333333, 1., 0.66666667, 1.66666667, 0.6, 0., 0.33333333, 0.33333333};
    int ndim = 2;
    idx_t nb_cols = 4;
    idx_t nb_rows = 6;
    seq_t c[nb_cols * ndim];
    for (idx_t i=0; i<(nb_cols*ndim); i++) { // Copy first series
        c[i] = s1[i];
    }
    ba_t mask[bit_bytes(nb_rows)];
    for (int i=0; i<nb_rows; i++) {
        bit_set(mask, i);
    }
    DTWSettings settings = dtw_settings_default();
    dtw_dba_ptrs(s, nb_rows, lengths, c, nb_cols, mask, 0, ndim, &settings);
    for (int i=0; i<(nb_cols*ndim); i++) {
        cr_assert_float_eq(c[i], exp_avg[i], 0.001);
    }
}


//----------------------------------------------------
// MARK: AUXILIARY FUNCTIONS

Test(aux, test_length_overflow_noblock) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    DTWBlock block = dtw_block_empty();
    idx_t length, expected_length;
    idx_t nb_series;
    
    // https://en.cppreference.com/w/c/types/limits
//    printf("The max value for sizes, SIZE_MAX = %zu\n", SIZE_MAX);
    
    nb_series = floor(sqrt(idx_t_max));
    length = dtw_distances_length(&block, nb_series, nb_series);
    expected_length = nb_series*(nb_series-1)/2;
//    printf("nb_series = %zu / length = %zu / expected = %zu\n", nb_series, length, expected_length);
    cr_assert_eq(length, expected_length); // no overflow
    
    nb_series = floor(sqrt(idx_t_max)) + 1;
    length = dtw_distances_length(&block, nb_series, nb_series);
//    printf("nb_series = %zu / length = %zu / expected = %zu\n", nb_series, length, expected_length);
    cr_assert_eq(length, 0); // overflow detected
}

Test(aux, test_length_overflow_block) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    idx_t length, expected_length;
    idx_t nb_series;
    
    // https://en.cppreference.com/w/c/types/limits
    printf("The max value for sizes, SIZE_MAX = %zu\n", SIZE_MAX);
    
//    idx_t b = floor(sqrt(SIZE_MAX)) - 1; // C cannot deal with SIZE_MAX to sqrt correctly
//    nb_series = 2*b + 1;
//    DTWBlock block = {.rb=0, .re=b, .cb=b+1, .ce=2*b+1};
//    length = dtw_distances_length(&block, nb_series);
//    expected_length = b*b;
//    printf("nb_series = %zu / length = %zu / expected = %zu / b = %zu\n", nb_series, length, expected_length, b);
//    cr_assert_eq(length, expected_length); // no overflow
    
//    idx_t b = floor(sqrt(SIZE_MAX)) - 1; // C cannot deal with SIZE_MAX to sqrt correctly
//    nb_series = 2*b + 1;
//    DTWBlock block = {.rb=0, .re=b+1, .cb=b, .ce=2*b+1};
//    length = dtw_distances_length(&block, nb_series);
//    expected_length = 0;
//    printf("nb_series = %zu / length = %zu / expected = %zu / b = %zu\n", nb_series, length, expected_length, b);
//    cr_assert_eq(length, expected_length); // no overflow
    
    // Just a bit overflow
//    idx_t b = floor(sqrt(SIZE_MAX)) - 1; // C cannot deal with SIZE_MAX to sqrt correctly
//    nb_series = 2*b + 1;
//    DTWBlock block = {.rb=0, .re=b+2, .cb=b, .ce=2*b+1};
//    length = dtw_distances_length(&block, nb_series);
//    expected_length = 0;
//    printf("nb_series = %zu / length = %zu / expected = %zu / b = %zu\n", nb_series, length, expected_length, b);
//    cr_assert_eq(length, expected_length); // overflow
    
    // Maximal overflow, fastest test
    nb_series = SIZE_MAX;
    DTWBlock block = {.rb=0, .re=nb_series, .cb=0, .ce=nb_series, .triu=true};
    length = dtw_distances_length(&block, nb_series, nb_series);
    expected_length = 0;
    cr_assert_eq(length, expected_length); // overflow
}

Test(aux, test_length_nooverflow_block) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    idx_t length, expected_length;
    idx_t nb_series;
    
    idx_t b = 10;
    nb_series = 2*b + 1;
    DTWBlock block = {.rb=0, .re=b, .cb=b+1, .ce=2*b+1, .triu=true};
    length = dtw_distances_length(&block, nb_series, nb_series);
    expected_length = b*b;
//    printf("nb_series = %zu / length = %zu / expected = %zu / b = %zu\n", nb_series, length, expected_length, b);
    cr_assert_eq(length, expected_length); // no overflow
}
