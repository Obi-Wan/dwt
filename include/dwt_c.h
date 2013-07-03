/*
 * dwt.h
 *
 *  Created on: 04/lug/2013
 *      Author: ben
 */

#ifndef DWT_C_H_
#define DWT_C_H_

extern "C" {
  float *
  dwt_haar_f32(float * const vol, const int num_dims, const int * const dims,
      const int levels, const int direct, const int inplace);

  double *
  dwt_haar_f64(double * const vol, const int num_dims, const int * const dims,
      const int levels, const int direct, const int inplace);
}

#endif /* DWT_H_ */
