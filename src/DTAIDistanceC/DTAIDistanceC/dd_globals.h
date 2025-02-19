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

#endif /* globals_h */
