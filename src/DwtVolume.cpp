/*
 * DwtVolume.cpp
 *
 *  Created on: Jul 8, 2013
 *      Author: vigano
 */

#include "DwtVolume.h"

#include "DwtExceptionBuilder.h"

#include <cmath>

namespace dwt {

  INLINE size_t
  DwtMultiDimensional::compute_cum_prod(const vector<size_t> & elems) const
  {
    size_t cumprod = 1;
    for (size_t elem : elems) { cumprod *=  elem; }
    return cumprod;
  }

  size_t
  DwtMultiDimensional::size() const
  {
    return compute_cum_prod(dims);
  }

  void
  DwtMultiDimensional::check_dims(const size_t & num_dims, const size_t & level)
  {
    DwtExceptionBuilder builder;
    if (dims.size() != num_dims)
    {
      throw builder.build<DwtWrongArgumentException>(
          "Number of allowed dimensions: ", num_dims, ", Number available: ", dims.size());
    }
    const size_t multiple = pow(2, level);
    for(size_t dim : dims)
    {
      if ((dim % (multiple)) > 0)
      {
        throw builder.build<DwtWrongArgumentException>(
            "Dimensions should be multiples of ", multiple);
      }
    }
  }

}  // namespace dwt

