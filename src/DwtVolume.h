/*
 * DwtV	olume.h
 *
 *  Created on: Jul 5, 2013
 *      Author: vigano
 */

#ifndef DWTVOLUME_H_
#define DWTVOLUME_H_

#include "dwt_definitions.h"

#include "DwtMemoryManager.h"

#include <vector>

namespace dwt {

  class DwtMultiDimensional {
  protected:
    vector<size_t> dims;

    template<typename ... Dims>
    void
    add_dims(const size_t & dim, Dims ... dims);
    void
    add_dims(const size_t & dim) { this->dims.push_back(dim); }

    size_t
    compute_cum_prod(const vector<size_t> & elems) const;

  public:
    DwtMultiDimensional() = default;
    DwtMultiDimensional(const vector<size_t> & _dims) : dims(_dims) { }

    const vector<size_t> &
    get_dims() const { return dims; }

    size_t size() const;

    void
    check_dims(const size_t & num_dims, const size_t & level);
  };

  template<typename Type>
  class DwtVolume : public DwtMultiDimensional, public DwtContainer<Type> {
  protected:
    size_t pitch;

    constexpr size_t
    compute_pitch(const vector<size_t> & _dims) const
    {
#ifdef USE_VECTORIZATION
      return _dims.size()
          ? (((DWT_MEMORY_ALIGN - (_dims[0] * sizeof(Type) % DWT_MEMORY_ALIGN)) % DWT_MEMORY_ALIGN) / sizeof(Type) + _dims[0])
          : 0;
#else
      return _dims.size() ? _dims[0] : 0;
#endif
    }

  public:
    DwtVolume(Type * _data, const vector<size_t> & _dims, const size_t & _pitch = 0)
    : DwtMultiDimensional(_dims), DwtContainer<Type>(_data)
    , pitch(_pitch ? _pitch : compute_pitch(_dims))
    { }
    template<typename ... Dims>
    DwtVolume(Type * _data, Dims ... dims);
    DwtVolume(const vector<size_t> & _dims);

    DwtVolume<Type> *
    get_sub_volume(const vector<size_t> & lims);
    void
    get_sub_volume(DwtVolume<Type> & sub_vol);
    void
    set_sub_volume(const DwtVolume<Type> & sub_vol);

    const size_t &
    get_pitch() const { return pitch; }

    size_t size() const;
  };

} /* namespace dwt */

template<typename ... Dims>
void
dwt::DwtMultiDimensional::add_dims(const size_t & dim, Dims ... dims)
{
	this->dims.push_back(dim);
	this->add_dims(dims ...);
}

template<typename Type>
template<typename ... Dims>
dwt::DwtVolume<Type>::DwtVolume(Type * _data, Dims ... dims)
: dwt::DwtContainer<Type>(_data), pitch(0)
{
	this->add_dims(dims ...);
}

template<typename Type>
dwt::DwtVolume<Type>::DwtVolume(const vector<size_t> & _dims)
: DwtMultiDimensional(_dims), pitch(compute_pitch(_dims))
{
  this->data = DwtMemoryManager::get_memory<Type>(this->size());
}

template<typename Type>
dwt::DwtVolume<Type> *
dwt::DwtVolume<Type>::get_sub_volume(const vector<size_t> & lims)
{
  DwtVolume<Type> * out = new DwtVolume<Type>(lims);

  DwtMemoryManager copier;
  DwtMemoryManager::CopyProperties props(out->get_dims(), out->get_pitch(), this->dims, this->pitch);

  copier.DEFAULT(strided_3D_copy<Type>)(out->data, this->data, props);
  return out;
}

template<typename Type>
void
dwt::DwtVolume<Type>::get_sub_volume(dwt::DwtVolume<Type> & sub_vol)
{
  DwtMemoryManager copier;
  DwtMemoryManager::CopyProperties props(sub_vol.get_dims(), sub_vol.get_pitch(), this->dims, this->pitch);

  copier.DEFAULT(strided_3D_copy<Type>)(sub_vol.get_data(), this->data, props);
}

template<typename Type>
void
dwt::DwtVolume<Type>::set_sub_volume(const dwt::DwtVolume<Type> & sub_vol)
{
  DwtMemoryManager copier;
  DwtMemoryManager::CopyProperties props(this->dims, this->pitch, sub_vol.get_dims(), sub_vol.get_pitch());

  copier.DEFAULT(strided_3D_copy<Type>)(this->data, sub_vol.get_data(), props);
}

template<typename Type>
size_t
dwt::DwtVolume<Type>::size() const
{
  vector<size_t> pitched_dims = dims;
  pitched_dims[0] = pitch;
  return this->compute_cum_prod(pitched_dims);
}


#endif /* DWTVOLUME_H_ */
