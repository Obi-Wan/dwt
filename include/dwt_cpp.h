/*
 * dwt_cpp.h
 *
 *  Created on: 04/lug/2013
 *      Author: ben
 */

#ifndef DWT_CPP_H_
#define DWT_CPP_H_

#include <vector>
#include <string>

bool
dwt_haar(float * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, std::string & error_msg);

bool
dwt_haar(double * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, std::string & error_msg);

bool
dwt_haar(std::vector<float *> & vols, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, std::string & error_msg);

bool
dwt_haar(std::vector<double *> & vols, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const bool & direct, std::string & error_msg);


bool
dwt_haar_soft_threshold(float * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const float & thr, std::string & error_msg);

bool
dwt_haar_soft_threshold(double * const vol, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const double & thr, std::string & error_msg);

bool
dwt_haar_soft_threshold(std::vector<float *> & vols, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const float & thr, std::string & error_msg);

bool
dwt_haar_soft_threshold(std::vector<double *> & vols, const std::vector<unsigned int> & dims,
    const unsigned int & levels, const double & thr, std::string & error_msg);


#endif /* DWT_CPP_H_ */
