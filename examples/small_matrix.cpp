/*
 * small_matrix.cpp
 *
 *  Created on: 11/lug/2013
 *      Author: ben
 */

#include "dwt_cpp.h"
#include <cstdio>

using namespace std;

typedef float TestType;

vector<TestType> matrix = {
    10, 10,  8, 8, 5, 5, 3, 1,
    10, 12, 10, 8, 7, 8, 4, 2,
    12,  8,  7, 6, 5, 4, 3, 1,
    11, 11, 12, 8, 4, 2, 1, 1,
    16, 12, 12, 3, 1, 1, 1, 1,
    18, 20, 18, 9, 4, 3, 2, 1,
    20, 14, 10, 6, 4, 5, 2, 1,
    20, 17, 13, 9, 6, 3, 2, 2,

    10, 10,  8, 8, 5, 5, 3, 1,
    10, 12, 10, 8, 7, 8, 4, 2,
    12,  8, 19, 6, 5, 4, 3, 1,
    11, 11, 12, 8, 4, 2, 1, 1,
    16, 12, 12, 3, 1, 1, 9, 9,
    18, 20, 10, 9, 4, 3, 2, 1,
    20, 14, 10, 6, 4, 5, 2, 1,
    20, 17, 15, 9, 6, 3, 8, 2,

    10, 10,  8, 8, 5, 5, 3, 1,
     9, 12,  9, 8, 7, 8, 4, 2,
    12,  8,  7, 6, 5, 4, 3, 1,
    11, 11, 12, 8, 4, 2, 1, 1,
     9, 12, 12, 3, 1, 1, 9, 9,
    18, 20,  6, 9, 4, 3, 2, 1,
    20, 14, 10, 6, 4, 5, 2, 1,
    20, 17, 15, 9, 6, 3, 8, 5,

    10, 10,  8, 7, 5, 5, 2, 1,
     9, 12,  9, 7, 7, 8, 4, 2,
    12,  8,  7, 7, 5, 4, 4, 1,
    11, 11,  0, 7, 4, 2, 1, 1,
     9, 12, 12, 7, 1, 1, 9, 9,
    18, 20,  1, 7, 4, 3, 6, 1,
    20, 14,  1, 8, 4, 5, 7, 1,
    20, 17, 15, 9, 6, 3, 8, 5,
};

void
print_volume(const vector<TestType> & vol, const vector<unsigned int> & dims);

int
main()
{
  vector<unsigned int> dims;
  dims.push_back(8);
  dims.push_back(8);
  dims.push_back(4);
  vector<TestType> original_matrix(matrix.begin(), matrix.end());

  print_volume(matrix, dims);

  string error_msg_buff;
  if (!dwt_haar(matrix.data(), dims, 2, true, error_msg_buff))
  {
    printf("Error! %s\n", error_msg_buff.c_str());
  }

  print_volume(matrix, dims);

  if (!dwt_haar(matrix.data(), dims, 2, false, error_msg_buff))
  {
    printf("Error! %s\n", error_msg_buff.c_str());
  }

  for (size_t count = 0; count < matrix.size(); count++)
  {
    matrix[count] -= original_matrix[count];
  }
  print_volume(matrix, dims);
}

void
print_volume(const vector<TestType> & vol, const vector<unsigned int> & dims)
{
  printf("Printing first layer:\n");
  for(size_t count2 = 0; count2 < dims[2]; count2++)
  {
    for(size_t count1 = 0; count1 < dims[1]; count1++)
    {
      for(size_t count0 = 0; count0 < dims[0]; count0++)
      {
        printf(" %e", matrix[count2*8*8 + count1*8 + count0]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

