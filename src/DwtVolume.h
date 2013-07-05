/*
 * DwtVolume.h
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

	template<typename Type>
	class DwtVolume {
	protected:
		Type * data;

		vector<size_t> dims;

		template<typename ... Dims>
		void
		add_dims(const size_t & dim, Dims ... dims);
		void
		add_dims() { }

		size_t
		compute_cum_prod(const vector<size_t> & elems) const;
	public:
		template<typename ... Dims>
		DwtVolume(Type * _data, Dims ... dims);
		DwtVolume(Type * _data, const vector<size_t> & _dims)
		: data(_data), dims(_dims) { }
		DwtVolume(const vector<size_t> & _dims);

		virtual ~DwtVolume();

		size_t size() const;

		const Type *
		get_data() const { return data; }
		Type *
		get_data() { return data; }

		DwtVolume<Type> *
		get_sub_volume(const vector<size_t> & lims);
	};

} /* namespace dwt */

template<typename Type>
template<typename ... Dims>
void
dwt::DwtVolume<Type>::add_dims(const size_t & dim, Dims ... dims)
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
size_t
compute_cum_prod(const vector<size_t> & elems) const
{
	size_t cumprod = 1;
	for (size_t elem : elems) { cumprod *=  elem; }
	return cumprod;
}

template<typename Type>
size_t
dwt::DwtVolume<Type>::size() const
{
	return compute_cum_prod(dims);
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


#endif /* DWTVOLUME_H_ */
