/*
 * DwtMemoryManager.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: ben
 */

#include "DwtMemoryManager.h"

#include "DwtExceptionBuilder.h"

#include <cstdlib>

namespace dwt {

  DwtMemoryManager::CopyProperties::CopyProperties(
      const vector<size_t> & dest_dims, const size_t & pitch_dest,
      const vector<size_t> & src_dims, const size_t & pitch_src)
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
    this->src_skip.push_back(pitch_src);
    this->src_skip.insert(src_skip.end(), src_dims.begin()+1, src_dims.end()-1);
    this->dest_skip.push_back(pitch_dest);
    this->dest_skip.insert(dest_skip.end(), dest_dims.begin()+1, dest_dims.end()-1);
  }

  void *
  DwtMemoryManager::allocate(const size_t & num_bytes)
  {
    void * out = _mm_malloc(num_bytes, DWT_MEMORY_ALIGN);
    if (!out)
    {
      DwtExceptionBuilder exc_builder;
      throw exc_builder.build<DwtWrongArgumentException>(
                "Allocation of ", num_bytes, " bytes failed!");
    }
    return out;
  }

} /* namespace dwt */
