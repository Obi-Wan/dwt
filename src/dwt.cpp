/*
 * dwt.cpp
 *
 *  Created on: 04/lug/2013
 *      Author: ben
 */

#include "dwt_c.h"
#include "dwt_cpp.h"

#include "DwtTransform.h"
#include "dwt_exceptions.h"

bool
dwt_haar(float * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, std::string & error_msg)
{
  try {
    vector<size_t> converted_dims(dims.begin(), dims.end());
    auto volume = new dwt::DwtVolume<float>(vol, converted_dims);
    volume->check_dims(3, levels);
    dwt::DwtTransform<float> transform(volume->get_dims(), levels);
    if (direct) {
#pragma omp parallel
      transform.direct(*volume);
    } else {
#pragma omp parallel
      transform.inverse(*volume);
    }
    dwt::DwtMemoryManager::dispose_container(volume);
    return true;
  } catch (const dwt::DwtBasicException & e) {
    error_msg = string(e.what());
    return false;
  }
}

bool
dwt_haar(double * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, std::string & error_msg)
{
  try {
    vector<size_t> converted_dims(dims.begin(), dims.end());
    auto volume = new dwt::DwtVolume<double>(vol, converted_dims);
    volume->check_dims(3, levels);
    dwt::DwtTransform<double> transform(volume->get_dims(), levels);
    if (direct) {
#pragma omp parallel
      transform.direct(*volume);
    } else {
#pragma omp parallel
      transform.inverse(*volume);
    }
    dwt::DwtMemoryManager::dispose_container(volume);
    return true;
  } catch (const dwt::DwtBasicException & e) {
    error_msg = string(e.what());
    return false;
  }
}

bool
dwt_haar(std::vector<float *> & vols, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, std::string & error_msg)
{
  try {
    vector<size_t> converted_dims(dims.begin(), dims.end());
    dwt::DwtVolume<float> volume(NULL, converted_dims);
    volume.check_dims(3, levels);
    dwt::DwtTransform<float> transform(volume.get_dims(), levels);
#pragma omp parallel
    for(float * vol : vols)
    {
#pragma omp single
      volume.set_data(vol);
      if (direct) {
        transform.direct(volume);
      } else {
        transform.inverse(volume);
      }
    }
    volume.set_data(NULL);
    return true;
  } catch (const dwt::DwtBasicException & e) {
    error_msg = string(e.what());
    return false;
  }
}

bool
dwt_haar(std::vector<double *> & vols, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, std::string & error_msg)
{
  try {
    vector<size_t> converted_dims(dims.begin(), dims.end());
    dwt::DwtVolume<double> volume(NULL, converted_dims);
    volume.check_dims(3, levels);
    dwt::DwtTransform<double> transform(volume.get_dims(), levels);
#pragma omp parallel
    for(double * vol : vols)
    {
#pragma omp single
      volume.set_data(vol);
      if (direct) {
        transform.direct(volume);
      } else {
        transform.inverse(volume);
      }
    }
    volume.set_data(NULL);
    return true;
  } catch (const dwt::DwtBasicException & e) {
    error_msg = string(e.what());
    return false;
  }
}

