#pragma once

#include "types.h"

#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>

namespace grace
{

//-----------------------------------------------------------------------------
// Utilities for comparing x, y, z or w components of the CUDA vector types,
// int{2,3,4}, float{2,3,4} etc.
//-----------------------------------------------------------------------------

struct vec4_compare_x
{
      template<typename Vec4>
      GRACE_HOST_DEVICE bool operator()(const Vec4 a, const Vec4 b)
      {
          return a.x < b.x;
      }
};

struct vec4_compare_y
{
      template<typename Vec4>
      GRACE_HOST_DEVICE bool operator()(const Vec4 a, const Vec4 b)
      {
          return a.y < b.y;
      }
};

struct vec4_compare_z
{
      template<typename Vec4>
      GRACE_HOST_DEVICE bool operator()(const Vec4 a, const Vec4 b)
      {
          return a.z < b.z;
      }
};

struct vec4_compare_w
{
      template<typename Vec4>
      GRACE_HOST_DEVICE bool operator()(const Vec4 a, const Vec4 b)
      {
          return a.w < b.w;
      }
};

// If TIter points to on-device data, then it should be one of
//     Any thrust iterator accepted by thrust::minmax_element();
//     thrust::device_ptr<T>;
// where 'T' is identical to std::iterator_traits<TIter>::value_type.
// Alternatively, TIter may be a custom iterator, but that custom iterator
// MUST be dereferenceable on both the host _AND_ the device!
// For example, an iterator over float4 values with the dereference operator
// __host__ __device__ float4& Float4Iter::operator*()
// {
//     #ifdef __CUDA_ARCH__
//     *d_ptr;
//     #else
//     cudaMemcpy(&tmp, d_ptr, sizeof(float4), cudaMemcpyDeviceToHost);
//     return tmp;
//     #endif
// }
// Though note that the above is flawed in that *iter = x will not work as
// desired when executed on the host.
// comp should be one of grace::vec4_compare_{x, y, z, w}().
template <typename TIter, typename T, typename Comp>
GRACE_HOST void min_max(
    TIter data_iter,
    const size_t N,
    T* min,
    T* max,
    const Comp comp)
{
    thrust::pair<TIter, TIter> min_max;
    min_max = thrust::minmax_element(thrust::device, data_iter, data_iter + N,
                                     comp);
    *min = *min_max.first;
    *max = *min_max.second;
}

// Vec4Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for Vec4Iter in
// grace::min_max().
template <typename Vec4Iter, typename T>
GRACE_HOST void min_max_x(
    Vec4Iter data_iter,
    const size_t N,
    T* min_x,
    T* max_x)
{
    typedef typename std::iterator_traits<Vec4Iter>::value_type Vec4;

    Vec4 min, max;
    min_max(data_iter, N, &min, &max, vec4_compare_x());
    *min_x = min.x;
    *max_x = max.x;
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec4, typename T>
GRACE_HOST void min_max_x(
    const Vec4* d_data,
    const size_t N,
    T* min_x,
    T* max_x)
{
    min_max_x(thrust::device_ptr<const Vec4>(d_data), N, min_x, max_x);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec4Iter template over the const Vec4*
// when provided with a non-const Vec4*. This template is therefore needed to
// correctly handle non-const Vec4* pointers.
template <typename Vec4, typename T>
GRACE_HOST void min_max_x(
    Vec4* d_data,
    const size_t N,
    T* min_x,
    T* max_x)
{
    min_max_x(thrust::device_ptr<const Vec4>(d_data), N, min_x, max_x);
}

template <typename Vec4, typename T>
GRACE_HOST void min_max_x(
    const thrust::device_vector<Vec4>& d_data,
    T* min_x,
    T* max_x)
{
    min_max_x(d_data.begin(), d_data.size(), min_x, max_x);
}

template <typename Vec4, typename T>
GRACE_HOST void min_max_x(
    const thrust::host_vector<Vec4>& h_data,
    T* min_x,
    T* max_x)
{
    min_max_x(h_data.begin(), h_data.size(), min_x, max_x);
}

// Vec4Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for Vec4Iter in
// grace::min_max().
template <typename Vec4Iter, typename T>
GRACE_HOST void min_max_y(
    Vec4Iter data_iter,
    const size_t N,
    T* min_y,
    T* max_y)
{
    typedef typename std::iterator_traits<Vec4Iter>::value_type Vec4;

    Vec4 min, max;
    min_max(data_iter, N, &min, &max, vec4_compare_y());
    *min_y = min.y;
    *max_y = max.y;
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec4, typename T>
GRACE_HOST void min_max_y(
    const Vec4* d_data,
    const size_t N,
    T* min_y,
    T* max_y)
{
    min_max_y(thrust::device_ptr<const Vec4>(d_data), N, min_y, max_y);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec4Iter template over the const Vec4*
// when provided with a non-const Vec4*. This template is therefore needed to
// correctly handly non-const Vec4* pointers.
template <typename Vec4, typename T>
GRACE_HOST void min_max_y(
    Vec4* d_data,
    const size_t N,
    T* min_y,
    T* max_y)
{
    min_max_y(thrust::device_ptr<const Vec4>(d_data), N, min_y, max_y);
}

template <typename Vec4, typename T>
GRACE_HOST void min_max_y(
    const thrust::device_vector<Vec4>& d_data,
    T* min_y,
    T* max_y)
{
    min_max_y(d_data.begin(), d_data.size(), min_y, max_y);
}

template <typename Vec4, typename T>
GRACE_HOST void min_max_y(
    const thrust::host_vector<Vec4>& h_data,
    T* min_y,
    T* max_y)
{
    min_max_y(h_data.begin(), h_data.size(), min_y, max_y);
}

// Vec4Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for Vec4Iter in
// grace::min_max().
template <typename Vec4Iter, typename T>
GRACE_HOST void min_max_z(
    Vec4Iter data_iter,
    const size_t N,
    T* min_z,
    T* max_z)
{
    typedef typename std::iterator_traits<Vec4Iter>::value_type Vec4;

    Vec4 min, max;
    min_max(data_iter, N, &min, &max, vec4_compare_z());
    *min_z = min.z;
    *max_z = max.z;
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec4, typename T>
GRACE_HOST void min_max_z(
    const Vec4* d_data,
    const size_t N,
    T* min_z,
    T* max_z)
{
    min_max_z(thrust::device_ptr<const Vec4>(d_data), N, min_z, max_z);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec4Iter template over the const Vec4*
// when provided with a non-const Vec4*. This template is therefore needed to
// correctly handly non-const Vec4* pointers.
template <typename Vec4, typename T>
GRACE_HOST void min_max_z(
    Vec4* d_data,
    const size_t N,
    T* min_z,
    T* max_z)
{
    min_max_z(thrust::device_ptr<const Vec4>(d_data), N, min_z, max_z);
}

template <typename Vec4, typename T>
GRACE_HOST void min_max_z(
    const thrust::device_vector<Vec4>& d_data,
    T* min_z,
    T* max_z)
{
    min_max_z(d_data.begin(), d_data.size(), min_z, max_z);
}

template <typename Vec4, typename T>
GRACE_HOST void min_max_z(
    const thrust::host_vector<Vec4>& h_data,
    T* min_z,
    T* max_z)
{
    min_max_z(h_data.begin(), h_data.size(), min_z, max_z);
}

// Vec4Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for Vec4Iter in
// grace::min_max().
template <typename Vec4Iter, typename T>
GRACE_HOST void min_max_w(
    Vec4Iter data_iter,
    const size_t N,
    T* min_w,
    T* max_w)
{
    typedef typename std::iterator_traits<Vec4Iter>::value_type Vec4;

    Vec4 min, max;
    min_max(data_iter, N, &min, &max, vec4_compare_w());
    *min_w = min.w;
    *max_w = max.w;
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec4, typename T>
GRACE_HOST void min_max_w(
    const Vec4* d_data,
    const size_t N,
    T* min_w,
    T* max_w)
{
    min_max_w(thrust::device_ptr<const Vec4>(d_data), N, min_w, max_w);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec4Iter template over the const Vec4*
// when provided with a non-const Vec4*. This template is therefore needed to
// correctly handly non-const Vec4* pointers.
template <typename Vec4, typename T>
GRACE_HOST void min_max_w(
    Vec4* d_data,
    const size_t N,
    T* min_w,
    T* max_w)
{
    min_max_w(thrust::device_ptr<const Vec4>(d_data), N, min_w, max_w);
}

template <typename Vec4, typename T>
GRACE_HOST void min_max_w(
    const thrust::device_vector<Vec4>& d_data,
    T* min_w,
    T* max_w)
{
    min_max_w(d_data.begin(), d_data.size(), min_w, max_w);
}

template <typename Vec4, typename T>
GRACE_HOST void min_max_w(
    const thrust::host_vector<Vec4>& h_data,
    T* min_w,
    T* max_w)
{
    min_max_w(h_data.begin(), h_data.size(), min_w, max_w);
}

//-----------------------------------------------------------------------------
// Utilities for reading in Gadget-2 (type 1) files
//-----------------------------------------------------------------------------

GRACE_HOST void skip_spacer(std::ifstream& file) {
    int dummy;
    file.read((char*)&dummy, sizeof(dummy));
}

struct gadget_header
{
  int npart[6];
  double mass[6];
  // double time;
  // double redshift;
  // int flag_sfr;
  // int flag_feedback;
  // int npartTotal[6];
  // int flag_cooling;
  // int num_files;
  // double BoxSize;
  // double Omega0;
  // double OmegaLambda;
  // double HubbleParam;
  // char fill[256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8];   /* fills to 256 Bytes */
  char fill [256 - 6*4 - 6*8];
};

GRACE_HOST gadget_header read_gadget_header(std::ifstream& file) {
    gadget_header header;
    skip_spacer(file);
    file.read((char*)&header.npart, sizeof(int)*6);
    file.read((char*)&header.mass, sizeof(double)*6);
    file.read((char*)&header.fill, sizeof(header.fill));
    skip_spacer(file);
    return header;
}

GRACE_HOST void read_gadget_gas(std::ifstream& file,
                     thrust::host_vector<float4>& xyzh,
                     thrust::host_vector<unsigned int>& ID,
                     thrust::host_vector<float>& m,
                     thrust::host_vector<float>& rho)
{
    int i_dummy;
    float f_dummy;
    int N_gas, N_withmasses;

    // Skip to end of header.
    file.seekg(std::ios::beg);
    gadget_header header = read_gadget_header(file);

    /* ------ Gas particles have index 0 ------ */

    // Calculate particle number counts, and read in positions block.
    N_gas = header.npart[0];
    xyzh.resize(N_gas); m.resize(N_gas); rho.resize(N_gas); ID.resize(N_gas);

    N_withmasses = 0;
    skip_spacer(file);
    for(int i=0; i<6; i++) {
        if (header.mass[i] == 0)
            N_withmasses += header.npart[i];

        for(int n=0; n<header.npart[i]; n++) {
            // Save gas particle data only.
            if (i == 0) {
                file.read((char*)&xyzh[n].x, sizeof(float));
                file.read((char*)&xyzh[n].y, sizeof(float));
                file.read((char*)&xyzh[n].z, sizeof(float));
            }
            else {
                file.read((char*)&f_dummy, sizeof(float));
                file.read((char*)&f_dummy, sizeof(float));
                file.read((char*)&f_dummy, sizeof(float));
            }
        }
    }
    skip_spacer(file);

    // Velocities.
    skip_spacer(file);
    for(int i=0; i<6; i++) {
        for(int n=0; n<header.npart[i]; n++) {
            file.read((char*)&f_dummy, sizeof(float));
            file.read((char*)&f_dummy, sizeof(float));
            file.read((char*)&f_dummy, sizeof(float));
        }
    }
    skip_spacer(file);

    // IDs.
    skip_spacer(file);
    for(int i=0; i<6; i++) {
        for(int n=0; n<header.npart[i]; n++) {
            // Save gas particle data only.
            if (i == 0)
                file.read((char*)&ID[n], sizeof(int));
            else
                file.read((char*)&i_dummy, sizeof(int));
        }
    }
    skip_spacer(file);

    // Masses (optional).  Spacers only exist if the block exists.
    // Otherwise, all particles of a given type have equal mass, saved in the
    // header.
    if (N_withmasses > 0)
        skip_spacer(file);
    else
        thrust::fill(m.begin(), m.end(), header.mass[0]);
    for(int i=0; i<6; i++) {
        if (header.mass[i] == 0) {
            for (int n=0; n<header.npart[i]; n++) {
                file.read((char*)&f_dummy, sizeof(float));
            }
        }
    }
    if (N_withmasses > 0)
        skip_spacer(file);

    // Gas properties (optional).
    if (N_gas > 0)
    {
        // Internal energies.
        skip_spacer(file);
        for(int n=0; n<N_gas; n++) {
            file.read((char*)&f_dummy, sizeof(float));
        }
        skip_spacer(file);

        // Densities.
        skip_spacer(file);
        for (int n=0; n<N_gas; n++) {
            file.read((char*)&rho[n], sizeof(float));
        }
        skip_spacer(file);

        // Smoothing lengths.
        skip_spacer(file);
        for (int n=0; n<N_gas; n++) {
            file.read((char*)&xyzh[n].w, sizeof(float));
        }
        skip_spacer(file);
    }

    return;
}

} // namespace grace
