/*
 * DwtTransform.h
 *
 *  Created on: Jul 5, 2013
 *      Author: vigano
 */

#ifndef DWTTRANSFORM_H_
#define DWTTRANSFORM_H_

#include "dwt_definitions.h"

#include "DwtVolume.h"

#include <cmath>

namespace dwt {

  class DwtTransform {
    template<typename Type>
    void
    UNOPTIM(direct_dim_0)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
    template<typename Type>
    void
    UNOPTIM(direct_dim_1)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
    template<typename Type>
    void
    UNOPTIM(direct_dim_2)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);

    template<typename Type>
    void
    UNOPTIM(inverse_dim_0)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
    template<typename Type>
    void
    UNOPTIM(inverse_dim_1)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
    template<typename Type>
    void
    UNOPTIM(inverse_dim_2)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
  public:
    DwtTransform() = default;
    virtual
    ~DwtTransform() = default;

    template<typename Type>
    void
    direct(DwtVolume<Type> & vol, const size_t & levels);

    template<typename Type>
    void
    inverse(DwtVolume<Type> & vol, const size_t & levels);
  };

} /* namespace dwt */

template<typename Type>
void
dwt::DwtTransform::direct(DwtVolume<Type> & vol, const size_t & levels)
{
  vol.check_dims(3, levels);

  const vector<size_t> & dims = vol.get_dims();
  const size_t & num_dims = dims.size();
  for (size_t level = 0; level < levels; level++)
  {
    const size_t fraction = pow(2, level);
    vector<size_t> lims = dims;
    for (size_t & lim : lims) { lim /= fraction; }

    DwtVolume<Type> * subvol = vol.get_sub_volume(lims);
    DwtVolume<Type> * temp_subvol = new DwtVolume<Type>(lims);

    for (size_t dim = 0; dim < num_dims; dim++)
    {
      switch(dim) {
        case 0: {
          // Prepare DIM1
          DEFAULT(direct_dim_0)(*temp_subvol, *subvol);
          break;
        }
        case 1: {
          // Prepare DIM2
          DEFAULT(direct_dim_1)(*temp_subvol, *subvol);
          break;
        }
        case 2: {
          // Prepare DIM3
          DEFAULT(direct_dim_2)(*temp_subvol, *subvol);
          break;
        }
      }
      swap(subvol, temp_subvol);
    }
    vol.set_sub_volume(*subvol);

    delete subvol;
    delete temp_subvol;
  }
}

template<typename Type>
void
dwt::DwtTransform::inverse(DwtVolume<Type> & vol, const size_t & levels)
{
  vol.check_dims(3, levels);

  const vector<size_t> & dims = vol.get_dims();
  const size_t & num_dims = dims.size();
  for (size_t level = 0; level < levels; level++)
  {
    const size_t effective_level = levels - level - 1;

    const size_t fraction = pow(2, effective_level);
    vector<size_t> lims = dims;
    for (size_t & lim : lims) { lim /= fraction; }

    DwtVolume<Type> * subvol = vol.get_sub_volume(lims);
    DwtVolume<Type> * temp_subvol = new DwtVolume<Type>(lims);

    for (size_t dim = 0; dim < num_dims; dim++)
    {
      const size_t effective_dim = num_dims - dim - 1;
      switch(effective_dim) {
        case 0: {
          // Prepare DIM1
          DEFAULT(inverse_dim_0)(*temp_subvol, *subvol);
          break;
        }
        case 1: {
          // Prepare DIM2
          DEFAULT(inverse_dim_1)(*temp_subvol, *subvol);
          break;
        }
        case 2: {
          // Prepare DIM3
          DEFAULT(inverse_dim_2)(*temp_subvol, *subvol);
          break;
        }
      }
      swap(subvol, temp_subvol);
    }
    vol.set_sub_volume(*subvol);

    delete subvol;
    delete temp_subvol;
  }
}

template<typename Type>
void
dwt::DwtTransform::UNOPTIM(direct_dim_0)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
{
  const vector<size_t> dims = src.get_dims();

  const size_t & line_length = dims[0];
  const size_t & tot_lines = dims[1];
  const size_t & tot_areas = dims[2];

  const size_t area_length = line_length * tot_lines;

#pragma omp for
  for (size_t area_num = 0; area_num < tot_areas; area_num++)
  {
    const Type * const src_area = src.get_data() + area_length * area_num;
    Type * const dest_area = dest.get_data() + area_length * area_num;

    for (size_t line_num = 0; line_num < tot_lines; line_num++)
    {
      const Type * const src_line = src_area + line_length * line_num;
      Type * const dest_line = dest_area + line_length * line_num;
      Type * const dest_half_line = dest_line + line_length / 2;

      for (size_t src_pixel = 0, dest_pixel = 0; src_pixel < line_length;
          src_pixel += 2, dest_pixel++)
      {
        dest_line[dest_pixel] = (src_line[src_pixel] + src_line[src_pixel+1]) / COEFF;
        dest_half_line[dest_pixel] = (src_line[src_pixel] - src_line[src_pixel+1]) / COEFF;
      }
    }
  }
}

template<typename Type>
void
dwt::DwtTransform::UNOPTIM(inverse_dim_0)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
{
  const vector<size_t> dims = src.get_dims();

  const size_t & line_length = dims[0];
  const size_t & tot_lines = dims[1];
  const size_t & tot_areas = dims[2];

  const size_t area_length = line_length * tot_lines;

#pragma omp for
  for (size_t area_num = 0; area_num < tot_areas; area_num++)
  {
    const Type * const src_area = src.get_data() + area_length * area_num;
    Type * const dest_area = dest.get_data() + area_length * area_num;

    for(size_t line_num = 0; line_num < tot_lines; line_num++)
    {
      const Type * const src_line = src_area + line_length * line_num;
      const Type * const src_half_line = src_line + line_length / 2;
      Type * const dest_line = dest_area + line_length * line_num;

      for(size_t src_pixel = 0, dest_pixel = 0; src_pixel < line_length;
          src_pixel++, dest_pixel += 2)
      {
        dest_line[dest_pixel] = (src_line[src_pixel] + src_half_line[src_pixel]) / COEFF;
        dest_line[dest_pixel+1] = (src_line[src_pixel] - src_half_line[src_pixel]) / COEFF;
      }
    }
  }
}

template<typename Type>
void
dwt::DwtTransform::UNOPTIM(direct_dim_1)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
{
  const vector<size_t> dims = src.get_dims();

  const size_t & line_length = dims[0];
  const size_t & tot_lines = dims[1];
  const size_t & tot_areas = dims[2];

  const size_t area_length = line_length * tot_lines;

#pragma omp for
  for (size_t area_num = 0; area_num < tot_areas; area_num++)
  {
    const Type * const src_area = src.get_data() + area_length * area_num;
    Type * const dest_area = dest.get_data() + area_length * area_num;

    for(size_t line_num = 0; line_num < tot_lines; line_num += 2)
    {
      const Type * const src_line = src_area + line_length * line_num;
      const Type * const src_next_line = src_line + line_length;

      Type * const dest_line = dest_area + line_length * line_num / 2;
      Type * const dest_half_num_lines = dest_area + line_length * (line_num + tot_lines) / 2;

      for(size_t src_pixel = 0, dest_pixel = 0; src_pixel < line_length;
          src_pixel++, dest_pixel++)
      {
        dest_line[dest_pixel] = (src_line[src_pixel] + src_next_line[src_pixel]) / COEFF;
        dest_half_num_lines[dest_pixel] = (src_line[src_pixel] - src_next_line[src_pixel]) / COEFF;
      }
    }
  }
}

template<typename Type>
void
dwt::DwtTransform::UNOPTIM(inverse_dim_1)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
{
  const vector<size_t> dims = src.get_dims();

  const size_t & line_length = dims[0];
  const size_t & tot_lines = dims[1];
  const size_t & tot_areas = dims[2];

  const size_t area_length = line_length * tot_lines;

#pragma omp for
  for (size_t area_num = 0; area_num < tot_areas; area_num++)
  {
    const Type * const src_area = src.get_data() + area_length * area_num;
    Type * const dest_area = dest.get_data() + area_length * area_num;

    for(size_t line_num = 0; line_num < tot_lines; line_num += 2)
    {
      const Type * const src_line = src_area + line_length * line_num / 2;
      const Type * const src_half_num_lines = src_area + line_length * (line_num + tot_lines) / 2;

      Type * const dest_line = dest_area + line_length * line_num;
      Type * const dest_next_line = dest_line + line_length;

      for(size_t src_pixel = 0, dest_pixel = 0; src_pixel < line_length;
          src_pixel++, dest_pixel++)
      {
        dest_line[dest_pixel] = (src_line[src_pixel] + src_half_num_lines[src_pixel]) / COEFF;
        dest_next_line[dest_pixel] = (src_line[src_pixel] - src_half_num_lines[src_pixel]) / COEFF;
      }
    }
  }
}

template<typename Type>
void
dwt::DwtTransform::UNOPTIM(direct_dim_2)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
{
  const vector<size_t> dims = src.get_dims();

  const size_t & line_length = dims[0];
  const size_t & tot_lines = dims[1];
  const size_t & tot_areas = dims[2];

  const size_t area_length = line_length * tot_lines;

#pragma omp for
  for (size_t area_num = 0; area_num < tot_areas; area_num += 2)
  {
    const Type * const src_area = src.get_data() + area_length * area_num;
    const Type * const src_next_area = src_area + area_length;

    Type * const dest_area = dest.get_data() + area_length * area_num;
    Type * const dest_half_num_areas = dest.get_data() + area_length * (area_num + tot_areas) / 2;

    for(size_t src_pixel = 0, dest_pixel = 0; src_pixel < area_length;
        src_pixel++, dest_pixel++)
    {
      dest_area[dest_pixel] = (src_area[src_pixel] + src_next_area[src_pixel]) / COEFF;
      dest_half_num_areas[dest_pixel] = (src_area[src_pixel] - src_next_area[src_pixel]) / COEFF;
    }
  }
}

template<typename Type>
void
dwt::DwtTransform::UNOPTIM(inverse_dim_2)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
{
  const vector<size_t> dims = src.get_dims();

  const size_t & line_length = dims[0];
  const size_t & tot_lines = dims[1];
  const size_t & tot_areas = dims[2];

  const size_t area_length = line_length * tot_lines;

#pragma omp for
  for (size_t area_num = 0; area_num < tot_areas; area_num++)
  {
    const Type * const src_area = src.get_data() + area_length * area_num;
    const Type * const src_half_num_areas = src.get_data() + area_length * (area_num + tot_areas) / 2;

    Type * const dest_area = dest.get_data() + area_length * area_num;
    Type * const dest_next_area = dest_area + area_length;

    for(size_t src_pixel = 0, dest_pixel = 0; src_pixel < area_length;
        src_pixel++, dest_pixel++)
    {
      dest_area[dest_pixel] = (src_area[src_pixel] + src_half_num_areas[src_pixel]) / COEFF;
      dest_next_area[dest_pixel] = (src_area[src_pixel] - src_half_num_areas[src_pixel]) / COEFF;
    }
  }
}


#endif /* DWTTRANSFORM_H_ */
