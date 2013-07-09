/*
 * DwtMemoryManager.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: ben
 */

#include "DwtMemoryManager.h"

#include "DwtExceptionBuilder.h"

namespace dwt {

  DwtMemoryManager::CopyProperties::CopyProperties(const vector<size_t> & dest_dims, const vector<size_t> & src_dims)
  {
    DwtExceptionBuilder exc_builder;
    const size_t & num_dims = src_dims.size();
    if (src_dims.size() != dest_dims.size()) {
      throw exc_builder.build<DwtWrongArgumentException>(
          "Source and destination should have same number of dimensions, instead: ",
          "src ", src_dims.size(), ", dest ", dest_dims.size());
    }
    dims.resize(num_dims);
    for (size_t count = 0; count < num_dims; count++)
    {
      dims[count] = min(src_dims[count], dest_dims[count]);
    }
    this->src_skip = vector<size_t>(src_dims.begin(), src_dims.end()-1);
    this->dest_skip = vector<size_t>(dest_dims.begin(), dest_dims.end()-1);
  }

  void
  DwtMemoryManager::CopyProperties::check_3d() const
  {
    DwtExceptionBuilder exc_builder;
    if (this->dims.size() != 3)
    {
      throw exc_builder.build<DwtWrongArgumentException>(
          "Properties should be for 3 dimensions, but only ", dims.size(), " specified!");
    }
    if (this->dest_skip.size() != 2)
    {
      throw exc_builder.build<DwtWrongArgumentException>(
          "Properties for 3 dimensions, should have 2 destination skips but only ", dest_skip.size(), " specified!");
    }
    if (this->src_skip.size() != 2)
    {
      throw exc_builder.build<DwtWrongArgumentException>(
          "Properties for 3 dimensions, should have 2 source skips but only ", src_skip.size(), " specified!");
    }
  }

} /* namespace dwt */
