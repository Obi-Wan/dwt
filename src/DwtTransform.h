/*
 * DwtTransform.h
 *
 *  Created on: Jul 5, 2013
 *      Author: vigano
 */

#ifndef DWTTRANSFORM_H_
#define DWTTRANSFORM_H_

#include "dwt_definitions.h"

#include "DwtVolume.h"

#include <cmath>

namespace dwt {

  class DwtTransform {
  public:
    DwtTransform() = default;
    virtual
    ~DwtTransform() = default;

    template<typename Type>
    void
    direct(DwtVolume<Type> & in, const size_t & levels);
  };

} /* namespace dwt */

template<typename Type>
void
dwt::DwtTransform::direct(DwtVolume<Type> & in, const size_t & levels)
{
  in.check_dims(3, levels);

  const vector<size_t> & dims = in.get_dims();
  const size_t & num_dims = dims.size();
  for (size_t level = 0; level < levels; level++)
  {
    const size_t fraction = pow(2, level);
    vector<size_t> lims = dims;
    for (size_t & lim : lims) { lim /= fraction; }
    DwtVolume<Type> * subvol = in.get_sub_volume(lims);

    const size_t fraction_next = pow(2, level+1);
    for (size_t dim = 0; dim < num_dims; dim++)
    {
      const size_t offset = dims[dim] / fraction_next;
      switch(dim) {
        case 0: {
          // Prepare DIM1
          break;
        }
        case 1: {
          // Prepare DIM2
          break;
        }
        case 2: {
          // Prepare DIM3
          break;
        }
      }
    }


  }
}


#endif /* DWTTRANSFORM_H_ */
