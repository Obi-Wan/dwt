/*
 * dwt_cpp.h
 *
 *  Created on: 04/lug/2013
 *      Author: ben
 */

#ifndef DWT_CPP_H_
#define DWT_CPP_H_

#include <vector>

float *
dwt_haar_f32(float * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct);

double *
dwt_haar_f64(double * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct);

void
dwt_haar_f32(std::vector<float *> & vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, const bool & dispose_old);

void
dwt_haar_f64(std::vector<double *> & vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, const bool & dispose_old);


#endif /* DWT_CPP_H_ */
