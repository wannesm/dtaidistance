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
    size_t lengths[6*n];
    for (int i=0; i<6*n; i++) {
        lengths[i] = 9;
    }
    size_t rl = 6*n*(6*n - 1) / 2;
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
    seq_t s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    seq_t s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    DTWSettings settings = dtw_settings_default();
    settings.max_dist = 1.415;
    seq_t wps[10*10];
    seq_t d = dtw_warping_paths(wps, s1, 9, s2, 9, true, true, &settings);
    printf("d=%f\n", d);
    dtw_print_wps(wps, 9, 9);
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
    double result[dtw_distances_length(&block, 6, false)];
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
         size_t lengths[] = {9, 9, 9, 9, 9, 9};
         double result[5];
         DTWSettings settings = dtw_settings_default();
         DTWBlock block = {.rb=1, .re=4, .cb=3, .ce=5};
         for (int j=0; j<1000; j++) {
             printf("---START---\n");
             dtw_distances_ptrs_parallel(s, 6, lengths, result, &block, &settings);
             for (int i=0; i<5; i++) {
                 printf("%.4f ", result[i]);
             }
             printf("\n");
//             cr_assert_float_eq(result[0], 1.41421356, 0.001);
//             cr_assert_float_eq(result[1], 0.00000000, 0.001);
//             cr_assert_float_eq(result[2], 2.23606798, 0.001);
//             cr_assert_float_eq(result[3], 1.73205081, 0.001);
//             cr_assert_float_eq(result[4], 1.41421356, 0.001);
         }
}

int main(int argc, const char * argv[]) {
    printf("Benchmarking ...\n");
    time_t start_t, end_t;
    double diff_t;
    time(&start_t);
    
//    benchmark1();
//    benchmark2();
//    benchmark3();
//    benchmark4();
//    benchmark5();
//    benchmark6();
//    benchmark7();
    benchmark8();
    
    time(&end_t);
    diff_t = difftime(end_t, start_t);
    printf("Execution time = %f\n", diff_t);
    
    return 0;
}
