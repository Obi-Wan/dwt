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

      vector<size_t> src_pitch;
      vector<size_t> dest_pitch;

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
  const size_t & line_length = props.dims[0];
  const size_t & tot_lines = props.dims[1];
  const size_t & tot_areas = props.dims[2];

  const size_t & dest_pitch_1 = props.dest_pitch[0];
  const size_t & dest_pitch_2 = props.dest_pitch[1];

  const size_t & src_pitch_1 = props.src_pitch[0];
  const size_t & src_pitch_2 = props.src_pitch[1];

#pragma omp for
  for(size_t num_area = 0; num_area < tot_areas; num_area++)
  {
    const Type * const src_area = src + num_area * src_pitch_2 * src_pitch_1;
    Type * const dest_area = dest + num_area * dest_pitch_2 * dest_pitch_1;

    for(size_t num_line = 0; num_line < tot_lines; num_line++)
    {
      const Type * const src_line = src_area + num_line * src_pitch_1;
      Type * const dest_line = dest_area + num_line * dest_pitch_1;

      for(size_t pixel = 0; pixel < line_length; pixel++)
      {
        dest_line[pixel] = src_line[pixel];
      }
    }
  }
}

template<typename Type>
INLINE void
dwt::DwtMemoryManager::VECTORIZED(strided_3D_copy)(Type * const dest, const Type * const src, const CopyProperties & props)
{
  const size_t & line_length = props.dims[0];
  const size_t & tot_lines = props.dims[1];
  const size_t & tot_areas = props.dims[2];

  const size_t & dest_pitch_1 = props.dest_pitch[0];
  const size_t & dest_pitch_2 = props.dest_pitch[1];

  const size_t & src_pitch_1 = props.src_pitch[0];
  const size_t & src_pitch_2 = props.src_pitch[1];

  typedef float vVvf __attribute__((vector_size(DWT_MEMORY_ALIGN))) __attribute__((aligned(DWT_MEMORY_ALIGN)));
  const size_t unrolling = 8;
  const size_t shift = DWT_MEMORY_ALIGN / sizeof(Type);
  const size_t block = shift * unrolling;

  const size_t unroll_line_length = ROUND_DOWN(line_length, block);

#pragma omp for
  for(size_t num_area = 0; num_area < tot_areas; num_area++)
  {
    const Type * const src_area = src + num_area * src_pitch_2 * src_pitch_1;
    Type * const dest_area = dest + num_area * dest_pitch_2 * dest_pitch_1;

    for(size_t num_line = 0; num_line < tot_lines; num_line++)
    {
      const Type * const src_line = src_area + num_line * src_pitch_1;
      Type * const dest_line = dest_area + num_line * dest_pitch_1;

      for(size_t pixel = 0; pixel < unroll_line_length; pixel += block)
      {
        COPY_V(dest_line, src_line, pixel, 0);
        COPY_V(dest_line, src_line, pixel, 1);
        COPY_V(dest_line, src_line, pixel, 2);
        COPY_V(dest_line, src_line, pixel, 3);
        COPY_V(dest_line, src_line, pixel, 4);
        COPY_V(dest_line, src_line, pixel, 5);
        COPY_V(dest_line, src_line, pixel, 6);
        COPY_V(dest_line, src_line, pixel, 7);
      }

      for(size_t pixel = unroll_line_length; pixel < line_length; pixel++)
      {
        dest_line[pixel] = src_line[pixel];
      }
    }
  }
}

template<typename Type>
Type *
dwt::DwtMemoryManager::get_memory(size_t numel)
{
	void * out = NULL;
	posix_memalign(&out, DWT_MEMORY_ALIGN, numel * sizeof(Type));
	return (Type *) out;
}

template<typename Type>
Type *
dwt::DwtMemoryManager::dispose_container(dwt::DwtContainer<Type> * container)
{
  Type * out = container->get_data();

  container->set_data(NULL);
  delete container;

  return out;
}

#endif /* DWTMEMORYMANAGER_H_ */
