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

/* The sequence type type can be customized by changing the typedef. */
typedef double seq_t;

/*! The index type
 
 The advantage of using ssize_t instead of size_t is that this is
 compatible with the Microsoft Visual C implementation of OpenMP
 that requires a signed integer type for the variables in the loops.
 The disadvantage is that ssize_t is a POXIS standard definition and
 not a C standard library definition.
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

bool is_openmp_supported(void);

#endif /* globals_h */
