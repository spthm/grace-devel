#pragma once

#include "types.h"

#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>

// Wrap around all calls to CUDA functions to handle errors.
#define CUDA_HANDLE_ERR(code) {grace::cudaErrorCheck((code), __FILE__, __LINE__); }

namespace grace
{

GRACE_HOST void cudaErrorCheck(
    cudaError_t code,
    const char* file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error!\nMsg:  %s\nFile: %s @ line %d\n",
                cudaGetErrorString(code), file, line);

    if (abort)
        exit(code);
    }
}

//-----------------------------------------------------------------------------
// Utilities for generating random floats
//-----------------------------------------------------------------------------

// See:
// https://code.google.com/p/thrust/source/browse/examples/monte_carlo.cu
// as well as:
// http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
// and links therin.


// Thomas Wang hash.
GRACE_HOST_DEVICE unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

class random_float_functor
{
    const unsigned int offset;
    thrust::uniform_real_distribution<float> uniform;
    const unsigned int seed_factor;

public:
    random_float_functor() : offset(0u), seed_factor(1u),
                             uniform(0.0f, 1.0f) {}

    explicit random_float_functor(const unsigned int offset_) :
        offset(offset_), uniform(0.0f, 1.0f), seed_factor(1u) {}

    explicit random_float_functor(const float scale_) :
        offset(0u), uniform(0.0f, scale_), seed_factor(1u) {}

    explicit random_float_functor(const float low_,
                                  const float high_) :
        offset(0u), uniform(low_, high_), seed_factor(1u) {}

    explicit random_float_functor(const unsigned int offset_,
                                  const float low_,
                                  const float high_) :
        offset(offset_), uniform(low_, high_), seed_factor(1u) {}

    explicit random_float_functor(const unsigned int offset_,
                                  const unsigned int seed_factor_) :
        offset(offset_), uniform(0.0f, 1.0f), seed_factor(seed_factor_) {}

    explicit random_float_functor(const float low_,
                                  const float high_,
                                  const unsigned int seed_factor_) :
        offset(0u), uniform(low_, high_), seed_factor(seed_factor_) {}

    explicit random_float_functor(const unsigned int offset_,
                                  const float low_,
                                  const float high_,
                                  const unsigned int seed_factor_) :
        offset(offset_), uniform(low_, high_), seed_factor(seed_factor_) {}

    GRACE_HOST_DEVICE float operator() (unsigned int n)
    {
        unsigned int seed = n;
        for (unsigned int i=0; i<seed_factor; i++) {
            seed = hash(seed);
        }
        thrust::default_random_engine rng(seed);

        rng.discard(offset);

        return uniform(rng);
    }
};

class random_double_functor
{
    const unsigned int offset;
    thrust::uniform_real_distribution<double> uniform;
    const unsigned int seed_factor;

public:
    random_double_functor() : offset(0u), seed_factor(1u),
                             uniform(0.0, 1.0) {}

    explicit random_double_functor(const unsigned int offset_) :
        offset(offset_), uniform(0.0, 1.0), seed_factor(1u) {}

    explicit random_double_functor(const double scale_) :
        offset(0u), uniform(0.0, scale_), seed_factor(1u) {}

    explicit random_double_functor(const double low_,
                                   const double high_) :
        offset(0u), uniform(low_, high_), seed_factor(1u) {}

    explicit random_double_functor(const unsigned int offset_,
                                   const double low_,
                                   const double high_) :
        offset(offset_), uniform(low_, high_), seed_factor(1u) {}

    explicit random_double_functor(const unsigned int offset_,
                                   const unsigned int seed_factor_) :
        offset(offset_), uniform(0.0, 1.0), seed_factor(seed_factor_) {}

    explicit random_double_functor(const double low_,
                                   const double high_,
                                   const unsigned int seed_factor_) :
        offset(0u), uniform(low_, high_), seed_factor(seed_factor_) {}

    explicit random_double_functor(const unsigned int offset_,
                                   const double low_,
                                   const double high_,
                                   const unsigned int seed_factor_) :
        offset(offset_), uniform(low_, high_), seed_factor(seed_factor_) {}

    GRACE_HOST_DEVICE double operator() (unsigned int n)
    {
        unsigned int seed = n;
        for (unsigned int i=0; i<seed_factor; i++) {
            seed = hash(seed);
        }
        thrust::default_random_engine rng(seed);

        rng.discard(offset);

        return uniform(rng);
    }
};

class random_float4_functor
{
    float4 xyzw;
    unsigned int seed;
    thrust::uniform_real_distribution<float> xyz_uniform;
    thrust::uniform_real_distribution<float> w_uniform;
    const unsigned int seed_factor;

public:
    random_float4_functor() :
        xyz_uniform(0.0f, 1.0f),
        w_uniform(0.0f, 1.0f),
        seed_factor(1u) {}

    explicit random_float4_functor(const float xyz_low,
                                   const float xyz_high) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(0.0f, 1.0f),
        seed_factor(1u) {}

    explicit random_float4_functor(const float w_high) :
        xyz_uniform(0.0f, 1.0f),
        w_uniform(0.0f, w_high),
        seed_factor(1u) {}

    explicit random_float4_functor(const float xyz_low,
                                   const float xyz_high,
                                   const float w_high) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(0.0f, w_high),
        seed_factor(1u) {}

    explicit random_float4_functor(const float xyz_low,
                                   const float xyz_high,
                                   const float w_low,
                                   const float w_high) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(w_low, w_high),
        seed_factor(1u) {}

    explicit random_float4_functor(const float xyz_low,
                                   const float xyz_high,
                                   const unsigned int seed_factor_) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(0.0f, 1.0f),
        seed_factor(seed_factor_) {}

    explicit random_float4_functor(const float w_high,
                                   const unsigned int seed_factor_) :
        xyz_uniform(0.0f, 1.0f),
        w_uniform(0.0f, w_high),
        seed_factor(seed_factor_) {}

    explicit random_float4_functor(const float xyz_low,
                                   const float xyz_high,
                                   const float w_high,
                                   const unsigned int seed_factor_) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(0.0f, w_high),
        seed_factor(seed_factor_) {}

    explicit random_float4_functor(const float xyz_low,
                                   const float xyz_high,
                                   const float w_low,
                                   const float w_high,
                                   const unsigned int seed_factor_) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(w_low, w_high),
        seed_factor(seed_factor_) {}

    GRACE_HOST_DEVICE float4 operator() (unsigned int n)
    {
        seed = n;
        for (unsigned int i=0; i<seed_factor; i++) {
            seed = hash(seed);
        }
        thrust::default_random_engine rng(seed);

        xyzw.x = xyz_uniform(rng);
        xyzw.y = xyz_uniform(rng);
        xyzw.z = xyz_uniform(rng);
        xyzw.w = w_uniform(rng);

        return xyzw;
    }
};

class random_double4_functor
{
    double4 xyzw;
    unsigned int seed;
    thrust::uniform_real_distribution<double> xyz_uniform;
    thrust::uniform_real_distribution<double> w_uniform;
    const unsigned int seed_factor;

public:
    random_double4_functor() :
        xyz_uniform(0.0, 1.0),
        w_uniform(0.0, 1.0),
        seed_factor(1u) {}

    explicit random_double4_functor(const double xyz_low,
                                    const double xyz_high) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(0.0, 1.0),
        seed_factor(1u) {}

    explicit random_double4_functor(const double w_high) :
        xyz_uniform(0.0, 1.0),
        w_uniform(0.0, w_high),
        seed_factor(1u) {}

    explicit random_double4_functor(const double xyz_low,
                                    const double xyz_high,
                                    const double w_high) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(0.0, w_high),
        seed_factor(1u) {}

    explicit random_double4_functor(const double xyz_low,
                                    const double xyz_high,
                                    const double w_low,
                                    const double w_high) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(w_low, w_high),
        seed_factor(1u) {}

    explicit random_double4_functor(const double xyz_low,
                                    const double xyz_high,
                                    const unsigned int seed_factor_) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(0.0, 1.0),
        seed_factor(seed_factor_) {}

    explicit random_double4_functor(const double w_high,
                                    const unsigned int seed_factor_) :
        xyz_uniform(0.0, 1.0),
        w_uniform(0.0, w_high),
        seed_factor(seed_factor_) {}

    explicit random_double4_functor(const double xyz_low,
                                    const double xyz_high,
                                    const double w_high,
                                    const unsigned int seed_factor_) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(0.0, w_high),
        seed_factor(seed_factor_) {}

    explicit random_double4_functor(const double xyz_low,
                                    const double xyz_high,
                                    const double w_low,
                                    const double w_high,
                                    const unsigned int seed_factor_) :
        xyz_uniform(xyz_low, xyz_high),
        w_uniform(w_low, w_high),
        seed_factor(seed_factor_) {}

    GRACE_HOST_DEVICE double4 operator() (unsigned int n)
    {
        seed = n;
        for (unsigned int i=0; i<seed_factor; i++) {
            seed = hash(seed);
        }
        thrust::default_random_engine rng(seed);

        xyzw.x = xyz_uniform(rng);
        xyzw.y = xyz_uniform(rng);
        xyzw.z = xyz_uniform(rng);
        xyzw.w = w_uniform(rng);

        return xyzw;
    }
};

//-----------------------------------------------------------------------------
// Utilities for comparing x, y, z or w components of float4-like types
//-----------------------------------------------------------------------------

struct float4_compare_x
{
      template<typename Float4>
      GRACE_HOST_DEVICE bool operator()(const Float4 a, const Float4 b)
      {
          return a.x < b.x;
      }
};

struct float4_compare_y
{
      template<typename Float4>
      GRACE_HOST_DEVICE bool operator()(const Float4 a, const Float4 b)
      {
          return a.y < b.y;
      }
};

struct float4_compare_z
{
      template<typename Float4>
      GRACE_HOST_DEVICE bool operator()(const Float4 a, const Float4 b)
      {
          return a.z < b.z;
      }
};

struct float4_compare_w
{
      template<typename Float4>
      GRACE_HOST_DEVICE bool operator()(const Float4 a, const Float4 b)
      {
          return a.w < b.w;
      }
};

template <typename Float4, typename Float>
GRACE_HOST void min_max_x(
    Float* min_x,
    Float* max_x,
    const thrust::device_vector<Float4>& d_data)
{
    typedef typename thrust::device_vector<Float4>::const_iterator iter;
    thrust::pair<iter, iter> min_max;
    min_max = thrust::minmax_element(d_data.begin(), d_data.end(),
                                     float4_compare_x());
    *min_x = ((Float4) *min_max.first).x;
    *max_x = ((Float4) *min_max.second).x;
}

template <typename Float4, typename Float>
GRACE_HOST void min_max_y(
    Float* min_y,
    Float* max_y,
    const thrust::device_vector<Float4>& d_data)
{
    typedef typename thrust::device_vector<Float4>::const_iterator iter;
    thrust::pair<iter, iter> min_max;
    min_max = thrust::minmax_element(d_data.begin(), d_data.end(),
                                     float4_compare_y());
    *min_y = ((Float4) *min_max.first).y;
    *max_y = ((Float4) *min_max.second).y;
}

template <typename Float4, typename Float>
GRACE_HOST void min_max_z(
    Float* min_z,
    Float* max_z,
    const thrust::device_vector<Float4>& d_data)
{
    typedef typename thrust::device_vector<Float4>::const_iterator iter;
    thrust::pair<iter, iter> min_max;
    min_max = thrust::minmax_element(d_data.begin(), d_data.end(),
                                     float4_compare_z());
    *min_z = ((Float4) *min_max.first).z;
    *max_z = ((Float4) *min_max.second).z;
}

template <typename Float4, typename Float>
GRACE_HOST void min_max_w(
    Float* min_w,
    Float* max_w,
    const thrust::device_vector<Float4>& d_data)
{
    typedef typename thrust::device_vector<Float4>::const_iterator iter;
    thrust::pair<iter, iter> min_max;
    min_max = thrust::minmax_element(d_data.begin(), d_data.end(),
                                     float4_compare_w());
    *min_w = ((Float4) *min_max.first).w;
    *max_w = ((Float4) *min_max.second).w;
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
