/*
 * simd_operations.h
 *
 *  Created on: 16/lug/2013
 *      Author: ben
 */

#ifndef SIMD_OPERATIONS_H_
#define SIMD_OPERATIONS_H_

#include "dwt_definitions.h"

#include <immintrin.h>

namespace dwt {

template<typename Type>
class SIMDUnrolling {
public:
  const size_t shift;
  const size_t unrolling;
  const size_t block;

  typedef Type vVvf __attribute__((vector_size(DWT_MEMORY_ALIGN))) __attribute__((aligned(DWT_MEMORY_ALIGN)));

  SIMDUnrolling(const size_t & _unroll)
  : shift(DWT_MEMORY_ALIGN / sizeof(Type)), unrolling(_unroll)
  , block(shift * unrolling)
  { }

  constexpr size_t
  get_unroll(const size_t & tot_size) const
  {
    return ROUND_DOWN(tot_size, block);
  }

  void
  copy(Type * const dest, const Type * const src) const;
};

#ifdef __AVX__
template<>
void
SIMDUnrolling<float>::copy(float * const dest, const float * const src) const
{
  _mm256_store_ps(dest, _mm256_loadu_ps(src));
}

template<>
void
SIMDUnrolling<double>::copy(double * const dest, const double * const src) const
{
  _mm256_store_pd(dest, _mm256_loadu_pd(src));
}
#else
template<typename Type>
void
SIMDUnrolling<Type>::copy(Type * const dest, const Type * const src) const
{
  *(vVvf *)dest = *(vVvf *)src;
}
#endif


template<typename Type>
class Coeff {
public:
  typedef typename SIMDUnrolling<Type>::vVvf vVvf;

  static const vVvf get(const Type & coeff = COEFF);
};

template<>
class Coeff<float> {
public:
  typedef SIMDUnrolling<float>::vVvf vVvf;

  static const vVvf get(const float & coeff = COEFF)
  {
#if defined(__AVX__)
    return (const vVvf) _mm256_set1_ps(coeff);
#else
    return (const vVvf) _mm_set1_ps(coeff);
#endif
  }
};

template<>
class Coeff<double> {
public:
  typedef SIMDUnrolling<double>::vVvf vVvf;

  static const vVvf get(const double & coeff = COEFF)
  {
#if defined(__AVX__)
    return (const vVvf) _mm256_set1_pd(coeff);
#else
    return (const vVvf) _mm_set1_pd(coeff);
#endif
  }
};

template<typename Type>
class AccessAligned {
public:
  typedef typename SIMDUnrolling<Type>::vVvf vVvf;

  const vVvf load(const Type * const __restrict in) const
  {
    return *(vVvf *)in;
  }
  void store(Type * const __restrict out, const vVvf & in) const
  {
    *(vVvf *)out = in;
  }
};

template<typename Type>
class AccessStreamed : public AccessAligned<Type> {
public:
  typedef typename AccessAligned<Type>::vVvf vVvf;

  void store(Type * const __restrict out, const vVvf & in) const;
};

template<>
inline void
AccessStreamed<float>::store(float * const __restrict out,
    const AccessStreamed<float>::vVvf & in) const
{
#if defined(__AVX__)
  _mm256_stream_ps(out, in);
#else
  _mm_stream_ps(out, in);
#endif
}

template<>
inline void
AccessStreamed<double>::store(double * const __restrict out,
    const AccessStreamed<double>::vVvf & in) const
{
#if defined(__AVX__)
  _mm256_stream_pd(out, in);
#else
  _mm_stream_pd(out, in);
#endif
}

template<typename Type>
class AccessUnaligned {
public:
  typedef typename SIMDUnrolling<Type>::vVvf vVvf;

  const vVvf load(const Type * const __restrict in) const;
  void store(Type * const __restrict out, const vVvf & in) const;
};

template<>
class AccessUnaligned<double> {
public:
  typedef typename SIMDUnrolling<double>::vVvf vVvf;

  const vVvf load(const double * const in) const
  {
#if defined(__AVX__)
    return _mm256_loadu_pd(in);
#else
    return _mm_loadu_pd(in);
#endif
  }
  void store(double * const out, const vVvf & in) const
  {
#if defined(__AVX__)
    _mm256_storeu_pd(out, in);
#else
    _mm_storeu_pd(out, in);
#endif
  }
};

template<>
class AccessUnaligned<float> {
public:
  typedef typename SIMDUnrolling<float>::vVvf vVvf;

  const vVvf load(const float * const __restrict in) const
  {
#if defined(__AVX__)
    return _mm256_loadu_ps(in);
#else
    return _mm_loadu_ps(in);
#endif
  }
  void store(float * const __restrict out, const vVvf & in) const
  {
#if defined(__AVX__)
    _mm256_storeu_ps(out, in);
#else
    _mm_storeu_ps(out, in);
#endif
  }
};

template<class access, typename Type>
class OpDim0 {
public:
  typedef typename SIMDUnrolling<Type>::vVvf vVvf;

  OpDim0(const size_t & _shift) : coeff(Coeff<Type>::get()), shift(_shift), accessor() { }

  void
  core_dir(Type * const __restrict out1, Type * const __restrict out2, const Type * const __restrict in) const;
  void
  core_inv(Type * const __restrict out, const Type * const __restrict in1, const Type * const __restrict in2) const;

protected:
  const vVvf coeff;
  const size_t shift;

  const access accessor;
};

template<class access>
class OpDim0<access, float> {
public:
  typedef typename SIMDUnrolling<float>::vVvf vVvf;

  OpDim0(const size_t & _shift) : coeff(Coeff<float>::get()), shift(_shift), accessor() { }

  void
  core_dir(float * const __restrict out1, float * const __restrict out2, const float * const __restrict in) const
  {
    const vVvf inVec1 = *((const vVvf *)in);
    const vVvf inVec2 = *((const vVvf *)(in+shift));

#if defined(__AVX__)
    const vVvf shuffle1 = _mm256_permute2f128_pd(inVec1, inVec2, 0x20);
    const vVvf shuffle2 = _mm256_permute2f128_pd(inVec1, inVec2, 0x31);

    const vVvf outVec1 = _mm256_hadd_ps(shuffle1, shuffle2) * coeff;
    const vVvf outVec2 = _mm256_hsub_ps(shuffle1, shuffle2) * coeff;
#elif defined(__SSE3__)
    const vVvf outVec1 = _mm_hadd_ps(inVec1, inVec2) * coeff;
    const vVvf outVec2 = _mm_hsub_ps(inVec1, inVec2) * coeff;
#else
    const vVvf shuffled1 = _mm_shuffle_ps(inVec1, inVec2, _MM_SHUFFLE(2, 0, 2, 0));
    const vVvf shuffled2 = _mm_shuffle_ps(inVec1, inVec2, _MM_SHUFFLE(3, 1, 3, 1));

    const vVvf outVec1 = _mm_add_ps(shuffled1, shuffled2) * coeff;
    const vVvf outVec2 = _mm_sub_ps(shuffled1, shuffled2) * coeff;
#endif

    *((vVvf *)out1) = outVec1;
    accessor.store(out2, outVec2);
  }

  void
  core_inv(float * const __restrict out, const float * const __restrict in1, const float * const __restrict in2) const
  {
    const vVvf inVec1 = *((const vVvf *)in1);
    const vVvf inVec2 = accessor.load(in2);

#if defined(__AVX__)
    const vVvf vecAdd = _mm256_add_ps(inVec1, inVec2) * coeff;
    const vVvf vecSub = _mm256_sub_ps(inVec1, inVec2) * coeff;

    const vVvf shuffle1 = _mm256_permute2f128_ps(vecAdd, vecSub, 0x13);
    const vVvf shuffle2 = _mm256_permute2f128_ps(vecAdd, vecSub, 0x02);

    const vVvf outVec1 = _mm256_unpacklo_ps(shuffle1, shuffle2);
    const vVvf outVec2 = _mm256_unpackhi_ps(shuffle1, shuffle2);
#else
    const vVvf vecAdd = _mm_add_ps(inVec1, inVec2) * coeff;
    const vVvf vecSub = _mm_sub_ps(inVec1, inVec2) * coeff;

    const vVvf outVec1 = _mm_unpacklo_ps(vecAdd, vecSub);
    const vVvf outVec2 = _mm_unpackhi_ps(vecAdd, vecSub);
#endif

    *((vVvf *)out) = outVec1;
    *((vVvf *)(out+shift)) = outVec2;
  }
protected:
  const vVvf coeff;
  const size_t shift;

  const access accessor;
};

template<class access>
class OpDim0<access, double> {
public:
  typedef typename SIMDUnrolling<double>::vVvf vVvf;

  OpDim0(const size_t & _shift) : coeff(Coeff<double>::get()), shift(_shift), accessor() { }

  void
  core_dir(double * const __restrict out1, double * const __restrict out2, const double * const __restrict in) const
  {
    const vVvf inVec1 = *((const vVvf *)in);
    const vVvf inVec2 = *((const vVvf *)(in+shift));

#if defined(__AVX__)
    const vVvf shuffle1 = _mm256_permute2f128_pd(inVec1, inVec2, 0x20);
    const vVvf shuffle2 = _mm256_permute2f128_pd(inVec1, inVec2, 0x31);

    const vVvf outVec1 = _mm256_hadd_pd(shuffle1, shuffle2) * coeff;
    const vVvf outVec2 = _mm256_hsub_pd(shuffle1, shuffle2) * coeff;
#elif defined(__SSE3__)
    const vVvf outVec1 = _mm_hadd_pd(inVec1, inVec2) * coeff;
    const vVvf outVec2 = _mm_hsub_pd(inVec1, inVec2) * coeff;
#else
    const vVvf unpacked1 = _mm_unpacklo_pd(inVec1, inVec2);
    const vVvf unpacked2 = _mm_unpackhi_pd(inVec1, inVec2);

    const vVvf outVec1 = _mm_add_pd(unpacked1, unpacked2) * coeff;
    const vVvf outVec2 = _mm_sub_pd(unpacked1, unpacked2) * coeff;
#endif

    *((vVvf *)out1) = outVec1;
    accessor.store(out2, outVec2);
  }

  void
  core_inv(double * const __restrict out, const double * const __restrict in1, const double * const __restrict in2) const
  {
    const vVvf inVec1 = *((const vVvf *)in1);
    const vVvf inVec2 = accessor.load(in2);

#if defined(__AVX__)
    const vVvf vecAdd = _mm256_add_pd(inVec1, inVec2) * coeff;
    const vVvf vecSub = _mm256_sub_pd(inVec1, inVec2) * coeff;

    const vVvf unpacked1 = _mm256_unpacklo_pd(vecAdd, vecSub);
    const vVvf unpacked2 = _mm256_unpackhi_pd(vecAdd, vecSub);

    const vVvf outVec1 = _mm256_permute2f128_pd(unpacked1, unpacked2, 0x20);
    const vVvf outVec2 = _mm256_permute2f128_pd(unpacked1, unpacked2, 0x31);
#else
    const vVvf vecAdd = _mm_add_pd(inVec1, inVec2) * coeff;
    const vVvf vecSub = _mm_sub_pd(inVec1, inVec2) * coeff;

    const vVvf outVec1 = _mm_unpacklo_pd(vecAdd, vecSub);
    const vVvf outVec2 = _mm_unpackhi_pd(vecAdd, vecSub);
#endif

    *((vVvf *)out) = outVec1;
    *((vVvf *)(out+shift)) = outVec2;
  }

protected:
  const vVvf coeff;
  const size_t shift;

  const access accessor;
};

#define LOAD_V(in, counter, offset) \
  const vVvf var_##offset = *(vVvf *)&in[counter + offset * shift]

#define LOAD_2V(in0, in1, counter, offset) \
  const vVvf var_0_##offset = *(vVvf *)&in0[counter + offset * shift]; \
  const vVvf var_1_##offset = *(vVvf *)&in1[counter + offset * shift]

#define PROCESS(offset) \
  const vVvf res_0_##offset = (var_0_##offset + var_1_##offset) * coeff; \
  const vVvf res_1_##offset = (var_0_##offset - var_1_##offset) * coeff

#define PROCESS_0_DIR(op, offres, off1, off2) \
    const vVvf res_0_##offres = op.hadd_dir(var_##off1, var_##off2); \
    const vVvf res_1_##offres = op.hsub_dir(var_##off1, var_##off2)

#define PROCESS_0_IND(op, offres1, offres2, offset) \
    const vVvf res_##offres1 = op.hadd_ind(var_0_##offset, var_1_##offset); \
    const vVvf res_##offres2 = op.hsub_ind(var_0_##offset, var_1_##offset)

#define STORE_V(out, counter, offset) \
    *(vVvf *)&out[counter + offset * shift] = res_##offset

#define STORE_2V(out0, out1, counter, offset) \
    *(vVvf *)&out0[counter + offset * shift] = res_0_##offset; \
    *(vVvf *)&out1[counter + offset * shift] = res_1_##offset



template<typename Type>
class SoftThreshold {
public:
  typedef typename SIMDUnrolling<Type>::vVvf vVvf;

  SoftThreshold(const Type & _thr)
  : thr(Coeff<Type>::get(_thr)), abs_mask(Coeff<Type>::get(0))
  , sign_mask(Coeff<Type>::get(0))
  { }

  const vVvf
  operator()(const vVvf & in);
protected:
  const vVvf thr;
  const vVvf abs_mask;
  const vVvf sign_mask;
};

template<>
class SoftThreshold<float> {
public:
  typedef typename SIMDUnrolling<float>::vVvf vVvf;

  SoftThreshold(const float & _thr)
  : thr(Coeff<float>::get(_thr))
#ifdef __AVX__
  , abs_mask(_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)))
  , sign_mask(_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)))
#else
  , abs_mask(_mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)))
  , sign_mask(_mm_castsi128_ps(_mm_set1_epi32(0x80000000)))
#endif
  , inv_2(Coeff<float>::get(0.5))
  { }

  const vVvf
  operator()(const vVvf & in)
  {
#ifdef __AVX__
    const vVvf new_elem = _mm256_and_ps(in, abs_mask) - thr;
    const vVvf abs_new_elem = (new_elem + _mm256_and_ps(new_elem, abs_mask)) * inv_2;
    return _mm256_or_ps(abs_new_elem, _mm256_and_ps(in, sign_mask));
#else
    const vVvf new_elem = _mm_and_ps(in, abs_mask) - thr;
    const vVvf abs_new_elem = (new_elem + _mm_and_ps(new_elem, abs_mask)) * inv_2;
    return _mm_or_ps(abs_new_elem, _mm_and_ps(in, sign_mask));
#endif
  }
protected:
  const vVvf thr;
  const vVvf abs_mask;
  const vVvf sign_mask;
  const vVvf inv_2;
};

template<>
class SoftThreshold<double> {
public:
  typedef typename SIMDUnrolling<double>::vVvf vVvf;

  SoftThreshold(const double & _thr)
  : thr(Coeff<double>::get(_thr))
#ifdef __AVX__
  , abs_mask(_mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffL)))
  , sign_mask(_mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000L)))
#else
  , abs_mask(_mm_castsi128_pd(_mm_set1_epi64x(0x7fffffffffffffffL)))
  , sign_mask(_mm_castsi128_pd(_mm_set1_epi64x(0x8000000000000000L)))
#endif
  , inv_2(Coeff<double>::get(0.5))
  { }

  const vVvf
  operator()(const vVvf & in)
  {
#ifdef __AVX__
    const vVvf new_elem = _mm256_and_pd(in, abs_mask) - thr;
    const vVvf abs_new_elem = (new_elem + _mm256_and_pd(new_elem, abs_mask)) * inv_2;
    return _mm256_or_pd(abs_new_elem, _mm256_and_pd(in, sign_mask));
#else
    const vVvf new_elem = _mm_and_pd(in, abs_mask) - thr;
    const vVvf abs_new_elem = (new_elem + _mm_and_pd(new_elem, abs_mask)) * inv_2;
    return _mm_or_pd(abs_new_elem, _mm_and_pd(in, sign_mask));
#endif
  }
protected:
  const vVvf thr;
  const vVvf abs_mask;
  const vVvf sign_mask;
  const vVvf inv_2;
};

}  // namespace dwt

#endif /* SIMD_OPERATIONS_H_ */
