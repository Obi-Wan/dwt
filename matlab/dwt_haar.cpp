/*
 * dwt_haar.cpp
 *
 *  Created on: 12/lug/2013
 *      Author: ben
 */

#include "mex.h"

#include "dwt_cpp.h"

#include <iterator>

using namespace std;

const static char error_id_wrong_arg[] = "dwt_haar:wrong_argument";
const static char error_id_num_args[] = "dwt_haar:not_enough_arguments";
const static char error_id_internal[] = "dwt_haar:internal_error";

const static char error_msg_num_dims[] = "Arrays need to be 3D";
const static char error_msg_size[] = "Arrays should all have the same size";

const static char error_msg_no_float[] = "Arrays need to be either singles or doubles";
const static char error_msg_diff_types[] = "Arrays should all have the same type";

const static char error_msg_num_args[] = "No array passed!";
const static char error_msg_second_arg[] = "Second argument should be a floating point number";
const static char error_msg_third_arg[] = "Third argument should be either a logical or a floating point number";


inline bool
check_size(vector<unsigned int> & dims, const mxArray * const input);
inline bool
check_type(mxClassID & type, const mxArray * const input);

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
  bool direct = true;
  mxClassID type = mxDOUBLE_CLASS;
  vector<unsigned int> dims;
  unsigned int levels = 1;
  string error_msg;

  if (nrhs < 1)
  {
    mexErrMsgIdAndTxt(error_id_num_args, error_msg_num_args);
    return;
  }
  if (nrhs >= 2)
  {
    switch (mxGetClassID(prhs[1]))
    {
    case mxDOUBLE_CLASS:
    {
      levels = mxGetScalar(prhs[1]);
      break;
    }
    case mxSINGLE_CLASS:
    {
      levels = *((float *)mxGetData(prhs[1]));
      break;
    }
    default:
    {
      mexErrMsgIdAndTxt(error_id_wrong_arg, error_msg_second_arg);
      return;
    }
    }
  }
  if (nrhs >= 3)
  {
    switch (mxGetClassID(prhs[2]))
    {
    case mxLOGICAL_CLASS:
    {
      direct = *mxGetLogicals(prhs[2]);
      break;
    }
    case mxDOUBLE_CLASS:
    {
      direct = mxGetScalar(prhs[2]);
      break;
    }
    case mxSINGLE_CLASS:
    {
      direct = *((float *)mxGetData(prhs[2]));
      break;
    }
    default:
    {
      mexErrMsgIdAndTxt(error_id_wrong_arg, error_msg_third_arg);
      return;
    }
    }
  }
  if (!check_size(dims, prhs[0]))
  {
    return;
  }
  if (!check_type(type, prhs[0]))
  {
    return;
  }
  if (mxIsCell(prhs[0]))
  {
    const mwSize tot_cells = mxGetNumberOfElements(prhs[0]);
    if (type == mxDOUBLE_CLASS)
    {
      vector<double *> arrays(tot_cells);
      for (mwIndex num_cell = 0; num_cell < tot_cells; num_cell++)
      {
        arrays[num_cell] = (double *)mxGetData(mxGetCell(prhs[0], num_cell));
      }
      if (!dwt_haar(arrays, dims, levels, direct, error_msg))
      {
        mexErrMsgIdAndTxt(error_id_internal, error_msg.c_str());
        return;
      }
    }
    else
    {
      vector<float *> arrays(tot_cells);
      for (mwIndex num_cell = 0; num_cell < tot_cells; num_cell++)
      {
        arrays[num_cell] = (float *)mxGetData(mxGetCell(prhs[0], num_cell));
      }
      if (!dwt_haar(arrays, dims, levels, direct, error_msg))
      {
        mexErrMsgIdAndTxt(error_id_internal, error_msg.c_str());
        return;
      }
    }
  }
  else
  {
    if (type == mxDOUBLE_CLASS)
    {
      if (!dwt_haar((double *)mxGetData(prhs[0]), dims, levels, direct, error_msg))
      {
        mexErrMsgIdAndTxt(error_id_internal, error_msg.c_str());
        return;
      }
    }
    else
    {
      if (!dwt_haar((float *)mxGetData(prhs[0]), dims, levels, direct, error_msg))
      {
        mexErrMsgIdAndTxt(error_id_internal, error_msg.c_str());
        return;
      }
    }
  }
  plhs[0] = (mxArray *) prhs[0];
}

bool
get_size(vector<unsigned int> & dims, const mxArray * const input)
{
  const mwSize & tot_dims = mxGetNumberOfDimensions(input);
  if (tot_dims != 3)
  {
    return false;
  }

  const mwSize * input_dims = mxGetDimensions(input);
  dims.resize(tot_dims);
  for (mwIndex num_dim = 0; num_dim < tot_dims; num_dim++)
  {
    dims[num_dim] = input_dims[num_dim];
  }
  return true;
}

bool
check_size(vector<unsigned int> & dims, const mxArray * const input)
{
  if (mxIsCell(input))
  {
    const mwSize & tot_cells = mxGetNumberOfElements(input);
    get_size(dims, mxGetCell(input, 0));
    vector<unsigned int> dims_temp;

    for (mwIndex num_cell = 1; num_cell < tot_cells; num_cell++)
    {
      if (!get_size(dims_temp, mxGetCell(input, num_cell)))
      {
        mexErrMsgIdAndTxt(error_id_wrong_arg, error_msg_num_dims);
        return false;
      }
      if (!equal(dims.begin(), dims.end(), dims_temp.begin()))
      {
        mexErrMsgIdAndTxt(error_id_wrong_arg, error_msg_size);
        return false;
      }
    }
  }
  else
  {
    if (!get_size(dims, input))
    {
      mexErrMsgIdAndTxt(error_id_wrong_arg, error_msg_num_dims);
      return false;
    }
  }
  return true;
}

bool
check_type(mxClassID & type, const mxArray * const input)
{
  if (mxIsCell(input))
  {
    const mwSize & tot_cells = mxGetNumberOfElements(input);
    type = mxGetClassID(mxGetCell(input, 0));
    if (type != mxSINGLE_CLASS || type != mxDOUBLE_CLASS)
    {
      mexErrMsgIdAndTxt(error_id_wrong_arg, error_msg_no_float);
      return false;
    }

    mxClassID type_temp;

    for (mwIndex num_cell = 1; num_cell < tot_cells; num_cell++)
    {
      type_temp = mxGetClassID(mxGetCell(input, num_cell));
      if (type_temp != mxSINGLE_CLASS || type_temp != mxDOUBLE_CLASS)
      {
        mexErrMsgIdAndTxt(error_id_wrong_arg, error_msg_no_float);
        return false;
      }
      if (type_temp != type)
      {
        mexErrMsgIdAndTxt(error_id_wrong_arg, error_msg_diff_types);
        return false;
      }
    }
  }
  else
  {
    type = mxGetClassID(input);
    if (type != mxSINGLE_CLASS || type != mxDOUBLE_CLASS)
    {
      mexErrMsgIdAndTxt(error_id_wrong_arg, error_msg_no_float);
      return false;
    }
  }
  return true;
}


