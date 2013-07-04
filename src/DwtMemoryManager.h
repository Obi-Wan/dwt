/*
 * DwtMemoryManager.h
 *
 *  Created on: Jul 4, 2013
 *      Author: ben
 */

#ifndef DWTMEMORYMANAGER_H_
#define DWTMEMORYMANAGER_H_

#include "dwt_definitions.h"

#include <vector>

namespace dwt {

  class DwtMemoryManager {
  /** Right now there's no real memory management: only static helper functions
   */
  public:
    struct CopyProperties {
      vector<size_t> dims;

      vector<size_t> src_skip;
      vector<size_t> dest_skip;

      void
      check_3d() const;
    };

  protected:
    template<typename Type, size_t vector_byte_size>
    void
    strided_3D_copy_vec(Type * const dest, const Type * const src, const CopyProperties & props);
  public:
    DwtMemoryManager() = default;
    virtual
    ~DwtMemoryManager() = default;

    template<typename Type>
    void
    SSE2(strided_3D_copy)(Type * const dest, const Type * const src, const CopyProperties & props);
    template<typename Type>
    void
    AVX(strided_3D_copy)(Type * const dest, const Type * const src, const CopyProperties & props);
  };

} /* namespace dwt */

template<typename Type, size_t vector_byte_size>
INLINE void
dwt::DwtMemoryManager::strided_3D_copy_vec(Type * const dest, const Type * const src, const CopyProperties & props)
{
  props.check_3d();

  const size_t & length_line = props.dims[0];
  const size_t & num_lines_1 = props.dims[1];
  const size_t & num_lines_2 = props.dims[2];

  const size_t & dest_skip_1 = props.dest_skip[0];
  const size_t & dest_skip_2 = props.dest_skip[1];

  const size_t & src_skip_1 = props.src_skip[0];
  const size_t & src_skip_2 = props.src_skip[1];

  typedef float vVvf __attribute__((vector_size(vector_byte_size))) __attribute__((aligned(vector_byte_size)));
  const size_t unrolling = 8;
  const size_t shift = vector_byte_size / sizeof(Type);
  const size_t block = shift * unrolling;

  const size_t unroll_length_line = ROUND_DOWN(length_line, block);

#pragma omp for
  for(size_t line2 = 0; line2 < num_lines_2; line2++)
  {
    const Type * const src_2 = src + line2 * src_skip_2;
    Type * const dest_2 = dest + line2 * dest_skip_2;

    for(size_t line1 = 0; line1 < num_lines_1; line1++)
    {
      const Type * const src_1 = src_2 + line1 * src_skip_1;
      Type * const dest_1 = dest_2 + line1 * dest_skip_1;

      for(size_t pixel = 0; pixel < unroll_length_line; pixel += block)
      {
        COPY_V(dest_1, src_1, pixel, 0);
        COPY_V(dest_1, src_1, pixel, 1);
        COPY_V(dest_1, src_1, pixel, 2);
        COPY_V(dest_1, src_1, pixel, 3);
        COPY_V(dest_1, src_1, pixel, 4);
        COPY_V(dest_1, src_1, pixel, 5);
        COPY_V(dest_1, src_1, pixel, 6);
        COPY_V(dest_1, src_1, pixel, 7);
      }

      for(size_t pixel = unroll_length_line; pixel < length_line; pixel++)
      {
        dest_1[pixel] = src_1[pixel];
      }
    }
  }
}

template<typename Type>
INLINE void
dwt::DwtMemoryManager::SSE2(strided_3D_copy)(Type * const dest, const Type * const src, const CopyProperties & props)
{
  strided_3D_copy_vec<Type, 16>(dest, src, props);
}

template<typename Type>
INLINE void
dwt::DwtMemoryManager::AVX(strided_3D_copy)(Type * const dest, const Type * const src, const CopyProperties & props)
{
  strided_3D_copy_vec<Type, 32>(dest, src, props);
}

#endif /* DWTMEMORYMANAGER_H_ */
