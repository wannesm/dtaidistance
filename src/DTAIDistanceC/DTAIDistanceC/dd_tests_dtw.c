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
    seq_t d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, true, settings);
    free(wps);
    return d;
}

enum functions {
    fn_dtw_distance,
    fn_dtw_warping_paths_distance
};

struct dtw_test_params {
//    DTWFnPtr fn;
    enum functions fn;
    DTWSettings settings;
    int id;
};

DTWFnPtr get_function(enum functions name) {
    switch (name) {
        case fn_dtw_distance:
            return dtw_distance;
            break;
        case fn_dtw_warping_paths_distance:
            return dtw_warping_paths_distance;
            break;
    }
    cr_assert(false);
    return dtw_distance;
}


ParameterizedTestParameters(dtw, test_series1) {
    static struct dtw_test_params params[] = {
        {.fn = fn_dtw_distance, .settings={.window=0}, .id=0},
        {.fn = fn_dtw_warping_paths_distance, .settings={.window=0}, .id=1},
        {.fn = fn_dtw_distance, .settings={.window=0, .use_pruning=true}, .id=2}
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
    double d = get_function(param->fn)(s1, 9, s2, 9, &param->settings);
//    printf("d=%f\n", d);
    cr_assert_float_eq(d, sqrt(2), 0.001);
}

ParameterizedTestParameters(dtw, test_series1_e) {
    static struct dtw_test_params params[] = {
        {.fn = fn_dtw_distance, .settings={.window=0, .inner_dist=1}, .id=0},
        {.fn = fn_dtw_warping_paths_distance, .settings={.window=0, .inner_dist=1}, .id=1},
        {.fn = fn_dtw_distance, .settings={.window=0, .inner_dist=1, .use_pruning=true}, .id=2}
    };
    idx_t nb_params = sizeof (params) / sizeof (struct dtw_test_params);
    return cr_make_param_array(struct dtw_test_params, params, nb_params);
}

ParameterizedTest(struct dtw_test_params *param, dtw, test_series1_e) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 1, 2, 0, 0, 0, 0, 0, 0};
    double d = get_function(param->fn)(s1, 9, s2, 9, &param->settings);
//    printf("d=%f\n", d);
    cr_assert_float_eq(d, 2, 0.001);
}


ParameterizedTestParameters(dtw, test_series2) {
    static struct dtw_test_params params[] = {
        {.fn = fn_dtw_distance, .settings={.window=0}, .id=0},
        {.fn = fn_dtw_warping_paths_distance, .settings={.window=0}, .id=1},
        {.fn = fn_dtw_distance, .settings={.window=0, .use_pruning=true}, .id=2},
        {.fn = fn_dtw_distance, .settings={.window=3}, .id=3},
        {.fn = fn_dtw_warping_paths_distance, .settings={.window=3}, .id=4},
        {.fn = fn_dtw_distance, .settings={.window=3, .use_pruning=true}, .id=5}
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
    double d = get_function(param->fn)(s1, 12, s2, 11, &param->settings);
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

ParameterizedTestParameters(dtw, test_e) {
    static struct dtw_test_params params[] = {
        {.fn = fn_dtw_distance, .settings={.window=0}, .id=0},
        {.fn = fn_dtw_warping_paths_distance, .settings={.window=0}, .id=1},
        {.fn = fn_dtw_distance, .settings={.window=0, .use_pruning=true, .max_dist=0.2}, .id=2},
        {.fn = fn_dtw_warping_paths_distance, .settings={.window=0, .use_pruning=true, .max_dist=0.2}, .id=3}
    };
    size_t nb_params = sizeof (params) / sizeof (struct dtw_test_params);
    return cr_make_param_array(struct dtw_test_params, params, nb_params);
}

ParameterizedTest(struct dtw_test_params *param, dtw, test_e) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    double s1[] = {5.005335029629605081e-01, 5.157722489130834864e-01, 4.804319657333316340e-01, 4.520537745752661318e-01, 4.867408184050183717e-01, 4.806534229629605415e-01, 4.530552579964135518e-01, 4.667067057333316171e-01, 4.567955137333316040e-01, 4.414902037333315876e-01, 4.240597964014319321e-01, 4.225263829008334970e-01, 4.030970017333316280e-01, 4.404482984865574768e-01, 3.852339312962939077e-01, 3.634947117333316435e-01, 3.861488867383516266e-01, 3.413363679008334928e-01, 3.451913457333316004e-01, 3.695692377333316680e-01, 3.434781337333315809e-01, 3.063217006568062506e-01, 2.845283817333316145e-01, 2.955394357333315791e-01, 3.151374838781335619e-01, 2.561411067352764026e-01, 2.301194263297469400e-01, 2.478605028202762184e-01, 1.972828198566299318e-01, 2.150545617333316228e-01, 2.232865857333316273e-01, 2.492665580680986370e-01, 2.144049374050155388e-01, 2.079081117333316520e-01, 1.879600957333316391e-01, 1.638555197333316227e-01, 1.425566689000865583e-01, 2.016327177333316067e-01, 2.290943870240647606e-01, 1.900932117333316296e-01, 1.503233018025057766e-01, 1.970833717333316248e-01, 1.999393777333316191e-01, 2.018818837333316019e-01, 2.554168153357214144e-01, 2.345002377333316179e-01, 2.407103957333316113e-01, 2.762874997333316096e-01, 3.059693477333316203e-01, 3.328774862341668528e-01, 3.583867537333316200e-01, 3.743879884050183016e-01, 4.266385131705089373e-01, 4.445410410742424712e-01, 4.642271795675002033e-01, 4.402678696630802357e-01, 4.814591396296271641e-01, 5.317886460815400840e-01, 5.548714817383517683e-01, 5.062713000716849709e-01, 5.431524597333317050e-01, 5.537961812962939323e-01, 5.720852595675002261e-01, 5.933977447347652534e-01, 5.845479257333316969e-01, 6.133363017333317568e-01, 6.276481431102108877e-01, 6.132085097333317414e-01, 5.922371597333316862e-01, 5.778388756463566089e-01};
    double s2[] = {5.584292601075275808e-01, 5.214504501075275522e-01, 4.877978901075275542e-01, 5.078206201075274873e-01, 4.769738701075275644e-01, 4.478925501075275428e-01, 4.242528301075275676e-01, 4.307546401075275644e-01, 4.370594201075275187e-01, 4.331284101075275617e-01, 4.810766301075275475e-01, 4.250942801075275335e-01, 3.973955801075275684e-01, 4.380910701075275693e-01, 3.786794801075275552e-01, 3.850050201075275180e-01, 3.576176301075275621e-01, 2.987050201075275302e-01, 3.377542001075275468e-01, 3.262601401075275187e-01, 3.278248801075275276e-01, 3.347294101075275474e-01, 3.222199801075275594e-01, 3.372712101075275304e-01, 2.526810801075275448e-01, 1.774206901075275622e-01, 2.384015601075275825e-01, 2.419624201075275816e-01, 1.694136001075275677e-01, 1.983933401075275715e-01, 2.272449101075275646e-01, 1.490059201075275563e-01, 1.416013701075275744e-01, 1.997542401075275698e-01, 1.791462801075275613e-01, 1.712680901075275819e-01, 1.851759601075275707e-01, 1.450854801075275591e-01, 1.041379601075275718e-01, 9.028068310752757064e-02, 1.358144301075275839e-01, 2.006444701075275616e-01, 2.003521501075275768e-01, 2.100136501075275663e-01, 2.521797401075275280e-01, 2.364524601075275734e-01, 2.236850301075275771e-01, 2.873612101075275205e-01, 3.358473801075275156e-01, 3.288144201075275386e-01, 3.195859301075275605e-01, 3.482947201075275445e-01, 4.032929801075275655e-01, 4.566962501075275682e-01, 5.173766201075274962e-01, 5.463256501075275384e-01, 5.172673701075275465e-01, 5.054312901075275200e-01, 5.344046101075274890e-01, 5.389180101075274898e-01, 5.188896901075275014e-01, 5.484243401075274971e-01, 5.899157901075275934e-01, 5.987863201075275255e-01, 6.357147701075275270e-01, 6.277379101075275525e-01, 5.519873201075274904e-01, 5.634240801075275362e-01, 6.307956401075275332e-01, 6.488636001075275272e-01};
    DTWFnPtr fn = get_function(param->fn);
    DTWSettings settings = param->settings;
    double d = fn(s1, 70, s2, 70, &settings);
//    printf("d=%f\n", d);
    cr_assert_float_eq(d, 0.19430270196116387, 0.001);
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
    dtw_settings_set_psi(2, &settings);
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
    dtw_settings_set_psi(2, &settings);
    double d = dtw_distance(s2, 40, s1, 40, &settings);
    cr_assert_float_eq(d, 0.0, 0.001);
    d = dtw_warping_paths_distance(s2, 40, s1, 40, &settings);
    cr_assert_float_eq(d, 0.0, 0.001);
}

Test(dtw_psi, test_a_c) {
    double s1[] = {0.00,0.48,0.84,1.00,0.91,0.60,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,-0.28,0.22,
                   0.66,0.94,0.99,0.80,0.41,-0.08,-0.54,-0.88,-1.00,-0.88,-0.54,-0.07,0.42,
                   0.80,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34,0.15,0.61};
    double s2[] = {-0.84,-0.48,0.00,0.48,0.84,1.00,0.91,0.60,0.14,-0.18,-0.76,-0.98,-0.99,-0.71,
                   -0.28,0.22,0.66,0.70,0.99,0.80,0.41,-0.08,-0.54,-1.02,-1.00,-0.88,-0.54,
                   -0.07,0.42,0.80,0.99,1.10,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34};
    DTWSettings settings = dtw_settings_default();
    dtw_settings_set_psi(2, &settings);
    settings.window = 25;
    double d = dtw_distance(s2, 40, s1, 40, &settings);
    cr_assert_float_eq(d, 0.287054, 0.001);
    d = dtw_warping_paths_distance(s2, 40, s1, 40, &settings);
    cr_assert_float_eq(d, 0.287054, 0.001);
}



//----------------------------------------------------
// MARK: WPS

Test(wps, test_b_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {0., 0.01, 0.,   0.01, 0., 0.,   0.,   0.01, 0.01, 0.02, 0.,  0.};
    double s2[] = {0., 0.02, 0.02, 0.,   0., 0.01, 0.01, 0.,   0.,   0.,   0.};
    DTWSettings settings = dtw_settings_default();
    settings.window = 3;
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 13*12);
    double d = dtw_warping_paths(wps, s1, 12, s2, 11, true, true, true, &settings);
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
    double d = dtw_warping_paths(wps, s1, 12, s2, 11, true, true, true, &settings);
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
    double d = dtw_warping_paths(wps, s1, 7, s2, 7, true, true, true, &settings);
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
    double d = dtw_warping_paths(wps, s1, 7, s2, 7, true, true, true, &settings);
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
    double d = dtw_warping_paths(wps, s1, 7, s2, 7, true, true, true, &settings);
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
    double d = dtw_warping_paths(wps, s1, 7, s2, 7, true, true, true, &settings);
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
    double d = dtw_warping_paths(wps, s1, 9, s2, 9, true, true, true, &settings);
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
    idx_t length_i;
    for (idx_t i=0; i<l1+l2; i++) {
        i1[i] = 0;
        i2[i] = 0;
    }
    dtw_warping_path(s1, l1, s2, l2, i1, i2, &length_i, &settings);
    
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

Test(wps, test_f_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {1., 2, 2, 4, 5, 5};
    int l1 = 6;
    double s2[] = {1., 2, 2, 4, 4, 4, 5};
    int l2 = 7;
    idx_t i1s[] = {5, 4, 3, 3, 3, 2, 1, 0};
    idx_t i2s[] = {6, 6, 5, 4, 3, 2, 1, 0};
    DTWSettings settings = dtw_settings_default();
    settings.window = 2;
    DTWWps p = dtw_wps_parts(l1, l2, &settings);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * p.length);

    double d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, true, &settings);
    cr_assert_float_eq(d, 0.00, 0.001);

    idx_t i1[8], i2[8];
    idx_t il;
    d = dtw_warping_path(s1, l1, s2, l2, i1, i2, &il, &settings);
    cr_assert_float_eq(d, 0.00, 0.001);
    cr_assert_eq(il, 8);
    for (int i=0; i<8; i++) {
        cr_assert_eq(i1[i], i1s[i]);
        cr_assert_eq(i2[i], i2s[i]);
    }
    
    d = dtw_distance(s1, l1, s2, l2, &settings);
    cr_assert_float_eq(d, 0.00, 0.001);
    free(wps);
    dtw_printprecision_reset();
}


//----------------------------------------------------
// MARK: WPS - PSI

Test(wps_psi, test_a_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    // Two sine waves
    double s1[] = {0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,-0.28,0.22,
                   0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,-0.07,0.42,
                   0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34,0.15,0.61};
    double s2[] = {-0.84,-0.48,0.,0.48,0.84,1.,0.91,0.6,0.14,-0.35,-0.76,-0.98,-0.96,-0.71,
                   -0.28,0.22,0.66,0.94,0.99,0.8,0.41,-0.08,-0.54,-0.88,-1.,-0.88,-0.54,
                  -0.07,0.42,0.8,0.99,0.93,0.65,0.21,-0.29,-0.71,-0.96,-0.98,-0.75,-0.34};
    DTWSettings settings = dtw_settings_default();
    dtw_settings_set_psi(2, &settings);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 41*41);
    double d = dtw_warping_paths(wps, s1, 40, s2, 40, true, true, true, &settings);
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
    dtw_settings_set_psi(2, &settings);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 41*41);
    double d = dtw_warping_paths(wps, s2, 40, s1, 40, true, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 0.0, 0.001);
}

Test(dtw_psi, test_b_a) {
    /* For l1 >>> l2, the overlap_left_ri was wrong (used length of l2
     * while it should be until the end when no window is used).
     */
    double s1[] = {-0.86271501, -1.32160597, -1.2307838, -0.97743775, -0.88183547, -0.71453147, -0.70975136, -0.65238999, -0.48508599, -0.40860416, -0.5567877, -0.39904393, -0.51854679, -0.51854679, -0.23652005, -0.21261948, 0.16978966, 0.21281068, 0.6573613, 1.28355626, 1.88585065, 1.565583, 1.40305912, 1.64206483, 1.8667302};
    double s2[] = {-0.87446789, 0.50009064, -1.43396157, 0.52081263, 1.28752619};
    DTWSettings settings = dtw_settings_default();
    settings.psi_1b = 0;
    settings.psi_1e = 0;
    settings.psi_2b = 5; // len(s2)
    settings.psi_2e = 5; // len(s2)
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * (25+1)*(5+1));
    double d = dtw_warping_paths(wps, s1, 25, s2, 5, true, true, true, &settings);
    free(wps);
    cr_assert_float_eq(d, 2.138, 0.001);
}

Test(dtw_psi, test_c_a) {
    // Example from http://www.timeseriesclassification.com/description.php?Dataset=TwoLeadECG
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {1.8896,-0.23712,-0.23712,-0.20134,-0.16556,-0.20134,-0.16556,-0.12978,-0.058224,0.013335,0.031225,0.10278,0.013335,-0.094004,-0.058224,-0.11189,-0.14767,-0.16556,-0.14767,-0.094004,-0.14767,-0.16556,-0.16556,-0.21923,-0.21923,-0.25501,-0.20134,-0.20134,-0.18345,-0.23712,-0.20134,-0.23712,-0.12978,-0.11189,-0.46969,-1.2747,-2.3481,-2.8133,-2.7775,-2.5986,-2.3839,-2.0082,-1.8651,-1.6146,-1.3463,-1.1495,-0.88115,-0.55914,-0.34446,-0.16556,-0.0045548,0.2459,0.53214,0.65737,0.71104,0.74682,0.76471,0.76471,0.80049,0.81838,0.87204,0.88993,0.97938,0.97938,1.0152,1.0867,1.1583,1.1762,1.212,1.2656,1.2656,1.2477,1.2656,1.1762,1.0867,0.99727,0.88993,0.74682,0.63948,0.58581,0.47847,0.38902};
    int l1 = 82;
    double s2[] = {1,0.93163,0.094486,0.094486,0.038006,0.080366,0.080366,0.052126,0.080366,0.12273,0.22157,0.29217,0.41925,0.48985,0.39101,0.39101,0.30629,0.24981,0.19333,0.080366,-0.0043544,-0.018474,-0.089075,-0.11731,-0.14555,-0.17379,-0.21615,-0.27263,-0.20203,-0.315,-0.25851,-0.17379,-0.28675,-0.24439,0.16509,-0.11731,-1.0069,-1.9812,-2.4895,-2.786,-2.9272,-2.4612,-2.0518,-1.8964,-1.8258,-1.7411,-1.6705,-1.2893,-0.99276,-0.65388,-0.37148,-0.30087,-0.046714,0.30629,0.53221,0.65929,0.65929,0.72989,0.74401,0.87109,0.89933,0.95581,0.96993,1.0546,1.1394,1.2523,1.2523,1.2947,1.3088,1.3512,1.2806,1.2806,1.1394,1.097,0.89933,0.72989,0.67341,0.54633,0.37689,0.23569,0.10861,0.080366,-0.074955};
    int l2 = 83;
    idx_t i1s[] = {81, 81, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 69, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 53, 52, 51, 50, 49, 48, 48, 47, 46, 45, 44, 43, 42, 42, 42, 41, 40, 39, 38, 37, 36, 36, 35, 34, 33, 32, 31, 31, 30, 29, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    idx_t i2s[] = {80, 79, 78, 77, 77, 76, 75, 74, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 64, 63, 63, 62, 61, 60, 59, 59, 58, 58, 58, 58, 57, 56, 55, 54, 53, 52, 52, 51, 50, 49, 48, 47, 47, 46, 45, 44, 43, 42, 41, 41, 40, 39, 38, 37, 36, 35, 35, 34, 33, 32, 31, 30, 29, 28, 28, 28, 27, 26, 26, 25, 25, 24, 23, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4, 4, 4, 4, 4, 3, 2, 1};
    idx_t ils = 102;
    DTWSettings settings = dtw_settings_default();
    settings.window = 5;
    dtw_settings_set_psi(2, &settings);
    DTWWps p = dtw_wps_parts(l1, l2, &settings);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * p.length);
    
    double d = dtw_distance(s1, l1, s2, l2, &settings);
    cr_assert_float_eq(d, 1.6773062101885274, 0.001);

    d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, true, &settings);
    cr_assert_float_eq(d, 1.6773062101885274, 0.001);

    idx_t i1[ils], i2[ils];
    idx_t il;
    d = dtw_warping_path(s1, l1, s2, l2, i1, i2, &il, &settings);
    cr_assert_float_eq(d, 1.6773062101885274, 0.001);
    cr_assert_eq(il, ils);
//    printf("path[:%zu] = [", il);
//    for (idx_t i=il-1; i>=0; i--) {
//        printf("(%zd, %zd), ", i1[i], i2[i]);
//    }
//    printf("]\n");
    for (int i=0; i<ils; i++) {
//        printf("i1[%i]: %zu ?= %zu  /  i2[%i]: %zu ?= %zu\n", i, i1[i], i1s[i], i, i2[i], i2s[i]);
        cr_assert_eq(i1[i], i1s[i]);
        cr_assert_eq(i2[i], i2s[i]);
    }
    free(wps);
    dtw_printprecision_reset();
}

Test(dtw_psi, test_d_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {1., 2, 0};
    int l1 = 3;
    double s2[] = {1., 0, 1, 2, 1, 0, 1, 0, 0, 0, 0};
    int l2 = 11;
    idx_t i1s[] = {2, 1, 0};
    idx_t i2s[] = {4, 3, 2};
    idx_t ils = 3;
    DTWSettings settings = dtw_settings_default();
    settings.psi_1b = 0;
    settings.psi_1e = 0;
    settings.psi_2b = l2;
    settings.psi_2e = l2;
    settings.penalty = 0.1;
    DTWWps p = dtw_wps_parts(l1, l2, &settings);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * p.length);
    
    double d = dtw_distance(s1, l1, s2, l2, &settings);
//    printf("d=%.2f\n", d);
    cr_assert_float_eq(d, 1.0, 0.001);

    d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, true, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);

    idx_t i1[ils], i2[ils];
    idx_t il;
    d = dtw_warping_path(s1, l1, s2, l2, i1, i2, &il, &settings);
    cr_assert_float_eq(d, 1.0, 0.001);
    cr_assert_eq(il, ils);
//    printf("path[:%zu] = [", il);
//    for (idx_t i=il-1; i>=0; i--) {
//        printf("(%zd, %zd), ", i1[i], i2[i]);
//    }
//    printf("]\n");
    for (int i=0; i<ils; i++) {
//        printf("i1[%i]: %zu ?= %zu\n", i, i1[i], i1s[i]);
        cr_assert_eq(i1[i], i1s[i]);
        cr_assert_eq(i2[i], i2s[i]);
    }
    free(wps);
    dtw_printprecision_reset();
}

Test(dtw_psi, test_d_b) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s1[] = {1., 2, 0};
    int l1 = 3;
    double s2[] = {1., 0, 1, 2, 1, 0, 1, 0, 0, 0, 0};
    int l2 = 11;
    seq_t ds = 1.019803902718557;
    idx_t i1s[] = {2,  2, 2, 2, 1, 0};
    idx_t i2s[] = {10, 9, 8, 7, 6, 6};
    idx_t ils = 6;
    DTWSettings settings = dtw_settings_default();
    settings.psi_1b = 0;
    settings.psi_1e = 0;
    settings.psi_2b = l2;
    settings.psi_2e = 0;
    settings.penalty = 0.1;
    DTWWps p = dtw_wps_parts(l1, l2, &settings);
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * p.length);
    
    double d = dtw_distance(s1, l1, s2, l2, &settings);
//    printf("d=%.2f\n", d);
    cr_assert_float_eq(d, ds, 0.001);

    d = dtw_warping_paths(wps, s1, l1, s2, l2, true, true, true, &settings);
    cr_assert_float_eq(d, ds, 0.001);

    idx_t i1[ils], i2[ils];
    idx_t il;
    d = dtw_warping_path(s1, l1, s2, l2, i1, i2, &il, &settings);
    cr_assert_float_eq(d, ds, 0.001);
    cr_assert_eq(il, ils);
//    printf("path[:%zu] = [", il);
//    for (idx_t i=il-1; i>=0; i--) {
//        printf("(%zd, %zd), ", i1[i], i2[i]);
//    }
//    printf("]\n");
    for (int i=0; i<ils; i++) {
//        printf("i1[%i]: %zu ?= %zu\n", i, i1[i], i1s[i]);
        cr_assert_eq(i1[i], i1s[i]);
        cr_assert_eq(i2[i], i2s[i]);
    }
    free(wps);
    dtw_printprecision_reset();
}


//----------------------------------------------------
// MARK: AFFINITY

Test(affinity, test_a) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    dtw_printprecision_set(6);
    double s[] = {0, -1, -1, 0, 1, 2, 1};
    DTWSettings settings = dtw_settings_default();
    settings.window = 0;
    settings.penalty = 1;
    seq_t * wps = (seq_t *)malloc(sizeof(seq_t) * 8*8);
    seq_t tau = 0.36787944117144233;
    seq_t delta = -0.7357588823428847;
    seq_t delta_factor = 0.5;
    seq_t gamma = 1;
    double d = dtw_warping_paths_affinity(wps, s, 7, s, 7, true, false, true, /*only_triu=*/false, gamma, tau, delta, delta_factor, &settings);
    cr_assert_float_eq(wps[10], 0.37, 0.01);
    free(wps);
    cr_assert_float_eq(d, 7.0, 0.01);
    dtw_printprecision_reset();
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
    double exp_avg[] = {0.3, 1.1666666666666667, 1.95, 2.5, 2.0, 2.05, 0.9666666666666667, 0.0, 0.0, 0.0};
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
    
    dtw_dba_matrix(s, nb_rows, nb_cols, c, nb_cols, mask, 0, 1, &settings);
    
    for (idx_t i=0; i<nb_cols; i++) {
        cr_assert_float_eq(c[i], exp_avg[i], 0.001);
    }
}

Test(dba, test_a_ptrs) {
    #ifdef SKIPALL
    cr_skip_test();
    #endif
    
    // 0.5,    | 1, 2, 3,   2.0, 2.1, 1.0, 0, 0, 0
    // 0.4, 0, | 1, 1.5, 1.9, 2.0, 0.9, 1, 0, 0
    // 0.3     |
    
    double s1[] = {0.5, 1, 2, 3, 2.0, 2.1, 1.0, 0, 0, 0};
    double s2[] = {0.4, 0, 1, 1.5, 1.9, 2.0, 0.9, 1, 0, 0};
    double **s = (double **)malloc(2 * sizeof(double *));
    s[0] = s1;
    s[1] = s2;
    
    double exp_avg[] = {0.3, 1.1666666666666667, 1.95, 2.5, 2.0, 2.05, 0.9666666666666667, 0.0, 0.0, 0.0};
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
        
    dtw_dba_ptrs(s, nb_rows, lengths, c, nb_cols, mask, 0, 1, &settings);
    
    for (idx_t i=0; i<nb_cols; i++) {
        cr_assert_float_eq(c[i], exp_avg[i], 0.001);
    }
    
    free(s);
}

//----------------------------------------------------
// MARK: BOUNDS

Test(bounds, test_keogh_lb_1) {
#ifdef SKIPALL
    cr_skip_test();
#endif
    int size=4;
    double ra1[] = {1., 2, 1, 3};
    double ra2[] = {3., 4, 3, 0};
    DTWSettings settings = dtw_settings_default();
    settings.window=2;
    double d = lb_keogh(ra1, size, ra2, size, &settings);
    cr_assert_float_eq(d, 2.23606797749979, 0.001);
}
