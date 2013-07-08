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
    add_dims() { }

    size_t
    compute_cum_prod(const vector<size_t> & elems) const;

  public:
    const vector<size_t> &
    get_dims() const { return dims; }

    size_t size() const;

    void
    check_dims(const size_t & num_dims, const size_t & level);
  };

  template<typename Type>
  class DwtVolume : public DwtMultiDimensional {
  protected:
    Type * data;

  public:
    template<typename ... Dims>
    DwtVolume(Type * _data, Dims ... dims);
    DwtVolume(Type * _data, const vector<size_t> & _dims)
    : data(_data), dims(_dims) { }
    DwtVolume(const vector<size_t> & _dims);

    virtual ~DwtVolume();

    const Type *
    get_data() const { return data; }
    Type *
    get_data() { return data; }

    DwtVolume<Type> *
    get_sub_volume(const vector<size_t> & lims);
    void
    set_sub_volume(const DwtVolume<Type> & sub_vol);
  };

} /* namespace dwt */

template<typename ... Dims>
void
dwt::DwtMultiDimensional::add_dims(const size_t & dim, Dims ... dims)
{
	this->dims.push_back(dim);
	add_dims(dims ...);
}

template<typename Type>
template<typename ... Dims>
dwt::DwtVolume<Type>::DwtVolume(Type * _data, Dims ... dims)
: data(_data)
{
	this->add_dims(dims ...);
}

template<typename Type>
dwt::DwtVolume<Type> *
dwt::DwtVolume<Type>::get_sub_volume(const vector<size_t> & lims)
{
  Type * out = DwtMemoryManager::get_memory<Type>(compute_cum_prod(lims));
  DwtMemoryManager copier;
  DwtMemoryManager::CopyProperties props(lims, this->dims);

  copier.DEFAULT(strided_3D_copy)(out, data, props);
  return new DwtVolume<Type>(out, lims);
}

template<typename Type>
void
dwt::DwtVolume<Type>::set_sub_volume(const dwt::DwtVolume<Type> & sub_vol)
{
  DwtMemoryManager copier;
  DwtMemoryManager::CopyProperties props(this->dims, sub_vol.get_dims());

  copier.DEFAULT(strided_3D_copy)(data, sub_vol, props);
}


#endif /* DWTVOLUME_H_ */
