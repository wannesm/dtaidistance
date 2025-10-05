/*!
@header globals.h
@brief DTAIDistance.globals : Global settings and typedefs

@author Wannes Meert
@copyright Copyright © 2020 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/

#ifndef globals_h
#define globals_h

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


/* The sequence type type can be customized by changing the typedef. */
typedef double seq_t;

/*! The index type
 
 The advantage of using ssize_t instead of size_t is that this is
 compatible with the Microsoft Visual C implementation of OpenMP
 that requires a signed integer type for the variables in the loops.
 The disadvantage is that ssize_t is a POXIS standard definition and
 not a C standard library definition.
 
 https://developercommunity.visualstudio.com/content/problem/822384/openmp-compiler-errors-introduced-after-visual-stu.html
 https://developercommunity.visualstudio.com/idea/539086/openmp-unsigned-typed-induction-variables-in-paral.html
 */
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif
typedef ssize_t idx_t;
#define idx_t_max PTRDIFF_MAX

//typedef size_t idx_t;
//#define idx_t_max SIZE_MAX

/**
 @var printPrecision
 @abstract Number of decimals to print when printing (partial) distances.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
static int printPrecision = 3;
static int printDigits = 7; // 3+4
static char printBuffer[20];
static char printFormat[5];
#pragma GCC diagnostic pop

// Step function
enum StepType_e {
    TypeI,
    TypeIII
};
typedef enum StepType_e StepType;

// Inner distance / local distance
enum InnerDist_e {
    SquaredEuclidean=0,
    Euclidean=1
};
typedef enum InnerDist_e InnerDist;

// Window type
enum WindowType_e {
    WindowDiagonal=0,
    WindowSlanted=1
};
typedef enum WindowType_e WindowType;

// Min and max macros
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ARGMIN(a, x, b, y) (((a) < (b)) ? (x) : (y))
#define MIN3(a, b, c) MIN(MIN(a, b), c)
#define MAX3(a, b, c) MAX(MAX(a, b), c)
#define ARGMIN3(a, b, c) (((a) < (b)) ? ((((a) < (c)) ? (0) : (2))) : ((((b) < (c)) ? (1) : (2))))

// Bit operations (based on 8 bit)
typedef unsigned char ba_t;
#define ba_size 8
#define bit_bytes(l)     ( ((l + (ba_size / 2)) / ba_size) )
#define bit_array(a,l)   ( unsigned char a[((l + (ba_size / 2)) / ba_size)]; for (int i=0; i<l; i++) {a[i]=0;} )
#define bit_set(a,i)     ( a[(i/ba_size)] |=  (1 << (i%ba_size)) )
#define bit_clear(a,i)   ( a[(i/ba_size)] &= ~(1 << (i%ba_size)) )
#define bit_test(a,i)    ( a[(i/ba_size)] &   (1 << (i%ba_size)) )

// Print numbers
void print_nb(seq_t value);
void print_nbs(seq_t* values, idx_t b, idx_t e);
void print_ch(char* string);

// Path
typedef struct {
    idx_t i;
    idx_t j;
} DDPathEntry;

typedef struct {
    DDPathEntry *array;
    idx_t length;
    idx_t capacity;
    seq_t distance;
} DDPath;

void dd_path_init(DDPath *a, idx_t initial_capacity);
void dd_path_insert(DDPath *a, idx_t i, idx_t j);
void dd_path_insert_wo_doubles(DDPath *a, idx_t i, idx_t j);
void dd_path_extend(DDPath *a, DDPath *b);
void dd_path_extend_wo_doubles(DDPath *a, DDPath *b, idx_t overlap);
void dd_path_extend_wo_overlap(DDPath *a, DDPath *b, idx_t overlap);
void dd_path_free(DDPath *a);
void dd_path_reverse(DDPath *a);
void dd_path_print(DDPath *path);

// Range
typedef struct {
    idx_t b;
    idx_t e;
} DDRange;

#endif /* globals_h */
