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

  template<typename Type>
  class DwtContainer {
  protected:
    Type * data;

  public:
    DwtContainer() : data(NULL) { }
    DwtContainer(Type * _data) : data(_data) { }
    virtual
    ~DwtContainer() { if (data) { free(data); } }

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
  private:
    static void *
    allocate(const size_t & num_bytes);
  public:
    struct CopyProperties {
      vector<size_t> dims;

      vector<size_t> src_skip;
      vector<size_t> dest_skip;

      CopyProperties() = default;
      CopyProperties(const CopyProperties & old) = default;
      CopyProperties(const vector<size_t> & dest_dims, const size_t & pitch_dest,
          const vector<size_t> & src_dims, const size_t & pitch_src);
    };

    DwtMemoryManager() = default;
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
    static void
    dispose_container(DwtContainer<Type> * container);
    template<typename Type>
    static Type *
    get_data_dispose_container(DwtContainer<Type> * container);
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

  const size_t & dest_pitch_1 = props.dest_skip[0];
  const size_t & dest_pitch_2 = props.dest_skip[1];

  const size_t & src_pitch_1 = props.src_skip[0];
  const size_t & src_pitch_2 = props.src_skip[1];

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

#define COPY_V(out, in, counter, offset) \
  *(vVvf *)&out[counter + (offset) * shift] = *(vVvf *)&in[counter + (offset) * shift]
#define COPY_V_4(out, in, counter, offset) \
  COPY_V(out, in, counter, offset+0); \
  COPY_V(out, in, counter, offset+1); \
  COPY_V(out, in, counter, offset+2); \
  COPY_V(out, in, counter, offset+3)
#define SAFE_DWT_MEMORY_ALIGN 16

template<typename Type>
INLINE void
dwt::DwtMemoryManager::VECTORIZED(strided_3D_copy)(Type * const dest, const Type * const src, const CopyProperties & props)
{
  const size_t & line_length = props.dims[0];
  const size_t & tot_lines = props.dims[1];
  const size_t & tot_areas = props.dims[2];

  const size_t & dest_pitch_1 = props.dest_skip[0];
  const size_t & dest_pitch_2 = props.dest_skip[1];

  const size_t & src_pitch_1 = props.src_skip[0];
  const size_t & src_pitch_2 = props.src_skip[1];

  typedef float vVvf __attribute__((vector_size(SAFE_DWT_MEMORY_ALIGN))) __attribute__((aligned(SAFE_DWT_MEMORY_ALIGN)));
  const size_t unrolling = 4;
  const size_t shift = SAFE_DWT_MEMORY_ALIGN / sizeof(Type);
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
        COPY_V_4(dest_line, src_line, pixel, 0);
      }

      for(size_t pixel = unroll_line_length; pixel < line_length; pixel++)
      {
        dest_line[pixel] = src_line[pixel];
      }
    }
  }
}
#undef COPY_V
#undef COPY_V_4
#undef SAFE_DWT_MEMORY_ALIGN

template<typename Type>
Type *
dwt::DwtMemoryManager::get_memory(size_t numel)
{
	return (Type *) DwtMemoryManager::allocate(numel * sizeof(Type));
}

template<typename Type>
void
dwt::DwtMemoryManager::dispose_container(dwt::DwtContainer<Type> * container)
{
  container->set_data(NULL);
  delete container;
}

template<typename Type>
Type *
dwt::DwtMemoryManager::get_data_dispose_container(dwt::DwtContainer<Type> * container)
{
  Type * out = container->get_data();

  container->set_data(NULL);
  delete container;

  return out;
}

#endif /* DWTMEMORYMANAGER_H_ */
