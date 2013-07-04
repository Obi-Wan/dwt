/*
 * DwtMemoryManager.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: ben
 */

#include "DwtMemoryManager.h"

#include "DwtExceptionBuilder.h"

namespace dwt {

  void
  dwt::DwtMemoryManager::CopyProperties::check_3d() const
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
