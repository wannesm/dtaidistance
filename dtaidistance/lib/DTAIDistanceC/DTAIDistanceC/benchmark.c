//
//  benchmark.c
//  DTAIDistanceC
//
//  Copyright © 2020 Wannes Meert.
//  Apache License, Version 2.0, see LICENSE for details.
//

#include <stdio.h>
#include <time.h>

#include "dtw.h"
#include "dtw_openmp.h"

void benchmark1() {
    int size=10000;
   double ra1[size], ra2[size];
   int i;
   for (i=0; i<size; i++) {
       ra1[i] = rand() % 10;
       ra2[i] = rand() % 10;
   }
   
   DTWSettings settings = dtw_default_settings();
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
    int lengths[6*n];
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
    DTWSettings settings = dtw_default_settings();
    DTWBlock block = dtw_empty_block();
    dtw_matrix_parallel(s, 6*n, lengths, result, &block, &settings);
//    for (int i=0; i<rl; i++) {
//        printf("%.2f ", result[i]);
//    }
//    printf("\n");
    free(result);
}


int main(int argc, const char * argv[]) {
    printf("Benchmarking ...\n");
    time_t start_t, end_t;
    double diff_t;
    time(&start_t);
    
//    benchmark1();
    benchmark2();
    
    time(&end_t);
    diff_t = difftime(end_t, start_t);
    printf("Execution time = %f\n", diff_t);
    
    return 0;
}
