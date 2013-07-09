/*
 * dwt.cpp
 *
 *  Created on: 04/lug/2013
 *      Author: ben
 */

#include "dwt_c.h"
#include "dwt_cpp.h"

#include "DwtTransform.h"

float *
dwt_haar_f32(float * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct)
{
  vector<size_t> converted_dims(dims.begin(), dims.end());
  auto volume = new dwt::DwtVolume<float>(vol, converted_dims);
  dwt::DwtTransform transform;
  if (direct) {
    transform.direct(*volume, levels);
  } else {
    transform.inverse(*volume, levels);
  }
  return dwt::DwtMemoryManager::dispose_container(volume);
}

double *
dwt_haar_f64(double * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct)
{
  vector<size_t> converted_dims(dims.begin(), dims.end());
  auto volume = new dwt::DwtVolume<double>(vol, converted_dims);
  dwt::DwtTransform transform;
  if (direct) {
    transform.direct(*volume, levels);
  } else {
    transform.inverse(*volume, levels);
  }
  return dwt::DwtMemoryManager::dispose_container(volume);
}

