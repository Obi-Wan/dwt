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

  template<typename Type>
  class DwtTransform {
  protected:
    vector<DwtVolume<Type> *> buffers;

    void
    UNOPTIM(direct_dim_0)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
    void
    UNOPTIM(direct_dim_1)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
    void
    UNOPTIM(direct_dim_2)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);

    void
    UNOPTIM(inverse_dim_0)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
    void
    UNOPTIM(inverse_dim_1)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
    void
    UNOPTIM(inverse_dim_2)(DwtVolume<Type> & dest, const DwtVolume<Type> & src);
  public:
    DwtTransform() = default;
    DwtTransform(const vector<size_t> & dims, const size_t & levels)
    { this->init(dims, levels); }
    virtual
    ~DwtTransform()
    { for (DwtVolume<Type> * buf : this->buffers) { delete buf; } }

    void
    init(const vector<size_t> & dims, const size_t & levels);

    void
    direct(DwtVolume<Type> & vol);
    void
    inverse(DwtVolume<Type> & vol);
  };

} /* namespace dwt */

template<typename Type>
void
dwt::DwtTransform<Type>::init(const vector<size_t> & dims, const size_t & levels)
{
  for (size_t level = 0; level < levels; level++)
  {
    const size_t fraction = pow(2, level);
    vector<size_t> lims = dims;
    for (size_t & lim : lims) { lim /= fraction; }

    this->buffers.push_back(new DwtVolume<Type>(lims));
    this->buffers.push_back(new DwtVolume<Type>(lims));
  }
}

template<typename Type>
void
dwt::DwtTransform<Type>::direct(DwtVolume<Type> & vol)
{
  const size_t levels = this->buffers.size() / 2;
  const vector<size_t> & dims = vol.get_dims();
  const size_t & num_dims = dims.size();
  for (size_t level = 0; level < levels; level++)
  {
    DwtVolume<Type> * subvol = this->buffers[2 * level];
    DwtVolume<Type> * temp_subvol = this->buffers[2 * level + 1];

    vol.get_sub_volume(*subvol);

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
#pragma omp single
      swap(subvol, temp_subvol);
    }
    vol.set_sub_volume(*subvol);
  }
}

template<typename Type>
void
dwt::DwtTransform<Type>::inverse(DwtVolume<Type> & vol)
{
  const size_t levels = this->buffers.size() / 2;
  const vector<size_t> & dims = vol.get_dims();
  const size_t & num_dims = dims.size();
  for (size_t level = 0; level < levels; level++)
  {
    const size_t effective_level = levels - level - 1;

    DwtVolume<Type> * subvol = this->buffers[2 * effective_level];
    DwtVolume<Type> * temp_subvol = this->buffers[2 * effective_level + 1];

    vol.get_sub_volume(*subvol);

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
#pragma omp single
      swap(subvol, temp_subvol);
    }
    vol.set_sub_volume(*subvol);
  }
}

template<typename Type>
void
dwt::DwtTransform<Type>::UNOPTIM(direct_dim_0)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
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
dwt::DwtTransform<Type>::UNOPTIM(inverse_dim_0)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
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
dwt::DwtTransform<Type>::UNOPTIM(direct_dim_1)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
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
dwt::DwtTransform<Type>::UNOPTIM(inverse_dim_1)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
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
dwt::DwtTransform<Type>::UNOPTIM(direct_dim_2)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
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
dwt::DwtTransform<Type>::UNOPTIM(inverse_dim_2)(DwtVolume<Type> & dest, const DwtVolume<Type> & src)
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
