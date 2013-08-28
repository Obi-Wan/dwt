/*
 * dwt_definitions.h
 *
 *  Created on: Jul 4, 2013
 *      Author: ben
 */

#ifndef DWT_DEFINITIONS_H_
#define DWT_DEFINITIONS_H_

#if defined(HAVE_CONFIG_H)
/* Autotools build */
# include "config.h"
#elif defined(_MSC_VER)
# if defined(_DEBUG)
#   define DEBUG
# endif
#else
/* Other, potentially unsupported building system */
# warning "Unknown building system, may cause unexpected behavior"
#endif

/* Message Reporting Level */

#ifdef DEBUG
# ifndef WARNING
#   define WARNING
# endif
#endif

#ifdef WARNING
# ifndef INFO
#   define INFO
# endif
#endif

#if defined(DEBUG) || defined(WARNING) || defined(INFO)
# include <cstdio>
using namespace std;

# ifndef __PRETTY_FUNCTION__
#   define __PRETTY_FUNCTION__ __FUNCTION__
# endif
#endif

/* Debug/Warning/Info helpers */

#ifdef DEBUG
# define INLINE
# define DebugPrintf( x ) do { printf("Debug: "); printf x; } while(0)
# define DebugPrintfNameAndLine(...) do { fprintf(stderr, "%s, at %u: ", __PRETTY_FUNCTION__, __LINE__); fprintf(stderr, __VA_ARGS__); } while(0)
# define DebugReportException( x ) printf("Debug: %s", x.what())
# define DEBUG_DECL( x ) x
# define DEBUG_CALL( x ) do { x; } while(0)
#else
# define INLINE inline
# define DebugPrintf( x )
# define DebugPrintfNameAndLine(...)
# define DebugReportException( x )
# define DEBUG_DECL( x )
# define DEBUG_CALL( x )
#endif


/* Let's take care of MS VisualStudio before 2010 that don't ship with stdint */
#if defined(_MSC_VER) && _MSC_VER < 1600
typedef __int8 int8_t;
typedef unsigned __int8 uint8_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
# include <stdint.h>
#endif

#define COEFF 0.707106781186547461715008466853760182857513427734375

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

#if defined(HAVE_OMP)
# include <omp.h>
#endif

#define AVX(x) avx_##x
#define SSE3(x) sse3_##x
#define SSE2(x) sse2_##x
#define UNOPTIM(x) unoptimized_##x
//#define VECTORIZED(x) vectorized_##x

//#define __SSE3__

#if defined(__AVX__)
# define VECTORIZED(x) AVX(x)
#elif defined(__SSE3__)
# define VECTORIZED(x) SSE3(x)
#else
# define VECTORIZED(x) SSE2(x)
#endif

#ifdef USE_VECTORIZATION
# define DEFAULT(x) VECTORIZED(x)
#else
# define DEFAULT(x) UNOPTIM(x)
#endif

#ifdef __AVX__
# define DWT_VECTOR_SIZE 32
#else
# define DWT_VECTOR_SIZE 16
#endif

#define DWT_MEMORY_ALIGN 64
#define DWT_SAFE_MEMORY_ALIGN 16

using namespace std;

#endif /* DWT_DEFINITIONS_H_ */
