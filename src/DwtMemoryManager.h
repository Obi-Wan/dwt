/*
 * DwtMemoryManager.h
 *
 *  Created on: Jul 4, 2013
 *      Author: ben
 */

#ifndef DWTMEMORYMANAGER_H_
#define DWTMEMORYMANAGER_H_

#include "dwt_definitions.h"

#include <cstdlib>

#include <vector>

namespace dwt {

  template<typename Type>
  class DwtContainer {
  protected:
    Type * data;

  public:
    DwtContainer() : data(NULL) { }
    DwtContainer(Type * _data) : data(_data) { }
    virtual
    ~DwtContainer() { delete data; }

    const Type *
    get_data() const { return data; }
    Type *
    get_data() { return data; }
    void
    set_data(Type * _data) { data = _data; }
  };

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

      CopyProperties() = default;
      CopyProperties(const CopyProperties & old) = default;
      CopyProperties(const vector<size_t> & dest_dims, const vector<size_t> & src_dims);
    };

    DwtMemoryManager() = default;
    virtual
    ~DwtMemoryManager() = default;

    template<typename Type>
    void
    VECTORIZED(strided_3D_copy)(Type * const dest, const Type * const src, const CopyProperties & props);
    template<typename Type>
    void
    UNOPTIM(strided_3D_copy)(Type * const dest, const Type * const src, const CopyProperties & props);

    template<typename Type>
    static Type *
    get_memory(size_t numel);

    template<typename Type>
    static Type *
    dispose_container(DwtContainer<Type> * container);
  };

} /* namespace dwt */

template<typename Type>
void
dwt::DwtMemoryManager::UNOPTIM(strided_3D_copy)(
    Type * const dest, const Type * const src, const CopyProperties & props)
{
  props.check_3d();

  const size_t & length_line = props.dims[0];
  const size_t & num_lines_1 = props.dims[1];
  const size_t & num_lines_2 = props.dims[2];

  const size_t & dest_skip_1 = props.dest_skip[0];
  const size_t & dest_skip_2 = props.dest_skip[1];

  const size_t & src_skip_1 = props.src_skip[0];
  const size_t & src_skip_2 = props.src_skip[1];

#pragma omp for
  for(size_t line2 = 0; line2 < num_lines_2; line2++)
  {
    const Type * const src_2 = src + line2 * src_skip_2;
    Type * const dest_2 = dest + line2 * dest_skip_2;

    for(size_t line1 = 0; line1 < num_lines_1; line1++)
    {
      const Type * const src_1 = src_2 + line1 * src_skip_1;
      Type * const dest_1 = dest_2 + line1 * dest_skip_1;

      for(size_t pixel = 0; pixel < length_line; pixel++)
      {
        dest_1[pixel] = src_1[pixel];
      }
    }
  }
}

template<typename Type>
INLINE void
dwt::DwtMemoryManager::VECTORIZED(strided_3D_copy)(Type * const dest, const Type * const src, const CopyProperties & props)
{
  props.check_3d();

  const size_t & length_line = props.dims[0];
  const size_t & num_lines_1 = props.dims[1];
  const size_t & num_lines_2 = props.dims[2];

  const size_t & dest_skip_1 = props.dest_skip[0];
  const size_t & dest_skip_2 = props.dest_skip[1];

  const size_t & src_skip_1 = props.src_skip[0];
  const size_t & src_skip_2 = props.src_skip[1];

  typedef float vVvf __attribute__((vector_size(DWT_MEMORY_ALIGN))) __attribute__((aligned(DWT_MEMORY_ALIGN)));
  const size_t unrolling = 8;
  const size_t shift = DWT_MEMORY_ALIGN / sizeof(Type);
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
static Type *
get_memory(size_t numel)
{
	Type * out = NULL;
	posix_memalign(out, DWT_MEMORY_ALIGN, numel);
	return out;
}

template<typename Type>
static Type *
dispose_container(dwt::DwtContainer<Type> * container)
{
  Type * out = container->get_data();

  container->set_data(NULL);
  delete container;

  return out;
}

#endif /* DWTMEMORYMANAGER_H_ */
