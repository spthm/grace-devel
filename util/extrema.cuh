#pragma once

#include "types.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>

#include <math.h>

namespace grace
{

//-----------------------------------------------------------------------------
// Utilities for comparing x, y, z or w components of the CUDA vector types,
// int{2,3,4}, float{2,3,4} etc.
//-----------------------------------------------------------------------------

template <typename Vec>
struct less_x
{
      GRACE_HOST_DEVICE bool operator()(const Vec a, const Vec b)
      {
          return a.x < b.x;
      }
};

template <typename Vec>
struct less_y
{
      GRACE_HOST_DEVICE bool operator()(const Vec a, const Vec b)
      {
          return a.y < b.y;
      }
};

// Vec must be a vec3 or vec4 type.
template <typename Vec>
struct less_z
{
      GRACE_HOST_DEVICE bool operator()(const Vec a, const Vec b)
      {
          return a.z < b.z;
      }
};

// Vec must be a vec4 type.
template <typename Vec>
struct less_w
{
      GRACE_HOST_DEVICE bool operator()(const Vec a, const Vec b)
      {
          return a.w < b.w;
      }
};

//-----------------------------------------------------------------------------
// Utilities for finding the minimum and maximum of all x, y, z and w components
// of the CUDA vector types, int{2,3,4}, float{2,3,4} etc.
//-----------------------------------------------------------------------------

template <typename Vec2>
struct min_xy
{
    GRACE_HOST_DEVICE Vec2 operator()(const Vec2 a, const Vec2 b)
    {
        Vec2 res;
        res.x = min(a.x, b.x);
        res.y = min(a.y, b.y);
        return res;
    }
};

template <typename Vec3>
struct min_xyz
{
    GRACE_HOST_DEVICE Vec3 operator()(const Vec3 a, const Vec3 b)
    {
        Vec3 res;
        res.x = min(a.x, b.x);
        res.y = min(a.y, b.y);
        res.z = min(a.z, b.z);
        return res;
    }
};

template <typename Vec4>
struct min_xyzw
{
    GRACE_HOST_DEVICE Vec4 operator()(const Vec4 a, const Vec4 b)
    {
        Vec4 res;
        res.x = min(a.x, b.x);
        res.y = min(a.y, b.y);
        res.z = min(a.z, b.z);
        res.w = min(a.w, b.w);
        return res;
    }
};

template <typename Vec2>
struct max_xy
{
    GRACE_HOST_DEVICE Vec2 operator()(const Vec2 a, const Vec2 b)
    {
        Vec2 res;
        res.x = max(a.x, b.x);
        res.y = max(a.y, b.y);
        return res;
    }
};

template <typename Vec3>
struct max_xyz
{
    GRACE_HOST_DEVICE Vec3 operator()(const Vec3 a, const Vec3 b)
    {
        Vec3 res;
        res.x = max(a.x, b.x);
        res.y = max(a.y, b.y);
        res.z = max(a.z, b.z);
        return res;
    }
};

template <typename Vec4>
struct max_xyzw
{
    GRACE_HOST_DEVICE Vec4 operator()(const Vec4 a, const Vec4 b)
    {
        Vec4 res;
        res.x = max(a.x, b.x);
        res.y = max(a.y, b.y);
        res.z = max(a.z, b.z);
        res.w = max(a.w, b.w);
        return res;
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
// comp should be one of grace::less_<T>{x, y, z, w}().
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

// VecIter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename VecIter, typename T>
GRACE_HOST void min_max_x(
    VecIter data_iter,
    const size_t N,
    T* min_x,
    T* max_x)
{
    typedef typename std::iterator_traits<VecIter>::value_type Vec;

    Vec mins, maxs;
    min_max(data_iter, N, &mins, &maxs, less_x<Vec>());
    *min_x = mins.x;
    *max_x = maxs.x;
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec, typename T>
GRACE_HOST void min_max_x(
    const Vec* d_data,
    const size_t N,
    T* min_x,
    T* max_x)
{
    min_max_x(thrust::device_ptr<const Vec>(d_data), N, min_x, max_x);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic VecIter template over the const Vec*
// when provided with a non-const Vec*. This template is therefore needed to
// correctly handle non-const Vec* pointers.
template <typename Vec, typename T>
GRACE_HOST void min_max_x(
    Vec* d_data,
    const size_t N,
    T* min_x,
    T* max_x)
{
    min_max_x(thrust::device_ptr<const Vec>(d_data), N, min_x, max_x);
}

template <typename Vec, typename T>
GRACE_HOST void min_max_x(
    const thrust::device_vector<Vec>& d_data,
    T* min_x,
    T* max_x)
{
    min_max_x(d_data.begin(), d_data.size(), min_x, max_x);
}

template <typename Vec, typename T>
GRACE_HOST void min_max_x(
    const thrust::host_vector<Vec>& h_data,
    T* min_x,
    T* max_x)
{
    min_max_x(h_data.begin(), h_data.size(), min_x, max_x);
}

// VecIter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename VecIter, typename T>
GRACE_HOST void min_max_y(
    VecIter data_iter,
    const size_t N,
    T* min_y,
    T* max_y)
{
    typedef typename std::iterator_traits<VecIter>::value_type Vec;

    Vec mins, maxs;
    min_max(data_iter, N, &mins, &maxs, less_y<Vec>());
    *min_y = mins.y;
    *max_y = maxs.y;
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec, typename T>
GRACE_HOST void min_max_y(
    const Vec* d_data,
    const size_t N,
    T* min_y,
    T* max_y)
{
    min_max_y(thrust::device_ptr<const Vec>(d_data), N, min_y, max_y);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic VecIter template over the const Vec*
// when provided with a non-const Vec*. This template is therefore needed to
// correctly handle non-const Vec* pointers.
template <typename Vec, typename T>
GRACE_HOST void min_max_y(
    Vec* d_data,
    const size_t N,
    T* min_y,
    T* max_y)
{
    min_max_y(thrust::device_ptr<const Vec>(d_data), N, min_y, max_y);
}

template <typename Vec, typename T>
GRACE_HOST void min_max_y(
    const thrust::device_vector<Vec>& d_data,
    T* min_y,
    T* max_y)
{
    min_max_y(d_data.begin(), d_data.size(), min_y, max_y);
}

template <typename Vec, typename T>
GRACE_HOST void min_max_y(
    const thrust::host_vector<Vec>& h_data,
    T* min_y,
    T* max_y)
{
    min_max_y(h_data.begin(), h_data.size(), min_y, max_y);
}

// VecIter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename VecIter, typename T>
GRACE_HOST void min_max_z(
    VecIter data_iter,
    const size_t N,
    T* min_z,
    T* max_z)
{
    typedef typename std::iterator_traits<VecIter>::value_type Vec;

    Vec mins, maxs;
    min_max(data_iter, N, &mins, &maxs, less_z<Vec>());
    *min_z = mins.z;
    *max_z = maxs.z;
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec, typename T>
GRACE_HOST void min_max_z(
    const Vec* d_data,
    const size_t N,
    T* min_z,
    T* max_z)
{
    min_max_z(thrust::device_ptr<const Vec>(d_data), N, min_z, max_z);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic VecIter template over the const Vec*
// when provided with a non-const Vec*. This template is therefore needed to
// correctly handle non-const Vec* pointers.
template <typename Vec, typename T>
GRACE_HOST void min_max_z(
    Vec* d_data,
    const size_t N,
    T* min_z,
    T* max_z)
{
    min_max_z(thrust::device_ptr<const Vec>(d_data), N, min_z, max_z);
}

template <typename Vec, typename T>
GRACE_HOST void min_max_z(
    const thrust::device_vector<Vec>& d_data,
    T* min_z,
    T* max_z)
{
    min_max_z(d_data.begin(), d_data.size(), min_z, max_z);
}

template <typename Vec, typename T>
GRACE_HOST void min_max_z(
    const thrust::host_vector<Vec>& h_data,
    T* min_z,
    T* max_z)
{
    min_max_z(h_data.begin(), h_data.size(), min_z, max_z);
}

// VecIter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename VecIter, typename T>
GRACE_HOST void min_max_w(
    VecIter data_iter,
    const size_t N,
    T* min_w,
    T* max_w)
{
    typedef typename std::iterator_traits<VecIter>::value_type Vec;

    Vec mins, maxs;
    min_max(data_iter, N, &mins, &maxs, less_w<Vec>());
    *min_w = mins.w;
    *max_w = maxs.w;
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec, typename T>
GRACE_HOST void min_max_w(
    const Vec* d_data,
    const size_t N,
    T* min_w,
    T* max_w)
{
    min_max_w(thrust::device_ptr<const Vec>(d_data), N, min_w, max_w);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic VecIter template over the const Vec*
// when provided with a non-const Vec*. This template is therefore needed to
// correctly handle non-const Vec* pointers.
template <typename Vec, typename T>
GRACE_HOST void min_max_w(
    Vec* d_data,
    const size_t N,
    T* min_w,
    T* max_w)
{
    min_max_w(thrust::device_ptr<const Vec>(d_data), N, min_w, max_w);
}

template <typename Vec, typename T>
GRACE_HOST void min_max_w(
    const thrust::device_vector<Vec>& d_data,
    T* min_w,
    T* max_w)
{
    min_max_w(d_data.begin(), d_data.size(), min_w, max_w);
}

template <typename Vec, typename T>
GRACE_HOST void min_max_w(
    const thrust::host_vector<Vec>& h_data,
    T* min_w,
    T* max_w)
{
    min_max_w(h_data.begin(), h_data.size(), min_w, max_w);
}

// Vec2Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename Vec2Iter, typename Vec2>
GRACE_HOST void min_vec2(
    Vec2Iter data_iter,
    const size_t N,
    Vec2* mins)
{
    // This may incurr a non-negligible overhead, but it is guaranteed to
    // produce a corrrect result for all Vec2-compatible types.
    const Vec2 init = data_iter[0];
    *mins = thrust::reduce(data_iter, data_iter + N, init, min_xy<Vec2>());
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec2>
GRACE_HOST void min_vec2(
    const Vec2* d_data,
    const size_t N,
    Vec2* mins)
{
    min_vec2(thrust::device_ptr<const Vec2>(d_data), N, mins);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec2Iter template over the const Vec2*
// when provided with a non-const Vec2*. This template is therefore needed to
// correctly handle non-const Vec2* pointers.
template <typename Vec2>
GRACE_HOST void min_vec2(
    Vec2* d_data,
    const size_t N,
    Vec2* mins)
{
    min_vec2(thrust::device_ptr<const Vec2>(d_data), N, mins);
}

template <typename Vec2>
GRACE_HOST void min_vec2(
    const thrust::device_vector<Vec2>& d_data,
    Vec2* mins)
{
    min_vec2(d_data.begin(), d_data.size(), mins);
}

template <typename Vec2>
GRACE_HOST void min_vec2(
    const thrust::host_vector<Vec2>& h_data,
    Vec2* mins)
{
    min_vec2(h_data.begin(), h_data.size(), mins);
}

// Vec3Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename Vec3Iter, typename Vec3>
GRACE_HOST void min_vec3(
    Vec3Iter data_iter,
    const size_t N,
    Vec3* mins)
{
    // This may incurr a non-negligible overhead, but it is guaranteed to
    // produce a corrrect result for all Vec3-compatible types.
    const Vec3 init = data_iter[0];
    *mins = thrust::reduce(data_iter, data_iter + N, init, min_xyz<Vec3>());
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec3>
GRACE_HOST void min_vec3(
    const Vec3* d_data,
    const size_t N,
    Vec3* mins)
{
    min_vec3(thrust::device_ptr<const Vec3>(d_data), N, mins);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec3Iter template over the const Vec3*
// when provided with a non-const Vec3*. This template is therefore needed to
// correctly handle non-const Vec3* pointers.
template <typename Vec3>
GRACE_HOST void min_vec3(
    Vec3* d_data,
    const size_t N,
    Vec3* mins)
{
    min_vec3(thrust::device_ptr<const Vec3>(d_data), N, mins);
}

template <typename Vec3>
GRACE_HOST void min_vec3(
    const thrust::device_vector<Vec3>& d_data,
    Vec3* mins)
{
    min_vec3(d_data.begin(), d_data.size(), mins);
}

template <typename Vec3>
GRACE_HOST void min_vec3(
    const thrust::host_vector<Vec3>& h_data,
    Vec3* mins)
{
    min_vec3(h_data.begin(), h_data.size(), mins);
}

// Vec4Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename Vec4Iter, typename Vec4>
GRACE_HOST void min_vec4(
    Vec4Iter data_iter,
    const size_t N,
    Vec4* mins)
{
    // This may incurr a non-negligible overhead, but it is guaranteed to
    // produce a corrrect result for all Vec4-compatible types.
    const Vec4 init = data_iter[0];
    *mins = thrust::reduce(data_iter, data_iter + N, init, min_xyzw<Vec4>());
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec4>
GRACE_HOST void min_vec4(
    const Vec4* d_data,
    const size_t N,
    Vec4* mins)
{
    min_vec4(thrust::device_ptr<const Vec4>(d_data), N, mins);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec4Iter template over the const Vec4*
// when provided with a non-const Vec4*. This template is therefore needed to
// correctly handle non-const Vec4* pointers.
template <typename Vec4>
GRACE_HOST void min_vec4(
    Vec4* d_data,
    const size_t N,
    Vec4* mins)
{
    min_vec4(thrust::device_ptr<const Vec4>(d_data), N, mins);
}

template <typename Vec4>
GRACE_HOST void min_vec4(
    const thrust::device_vector<Vec4>& d_data,
    Vec4* mins)
{
    min_vec4(d_data.begin(), d_data.size(), mins);
}

template <typename Vec4>
GRACE_HOST void min_vec4(
    const thrust::host_vector<Vec4>& h_data,
    Vec4* mins)
{
    min_vec4(h_data.begin(), h_data.size(), mins);
}

// Vec2Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename Vec2Iter, typename Vec2>
GRACE_HOST void max_vec2(
    Vec2Iter data_iter,
    const size_t N,
    Vec2* maxs)
{
    // This may incurr a non-negligible overhead, but it is guaranteed to
    // produce a corrrect result for all Vec2-compatible types.
    const Vec2 init = data_iter[0];
    *maxs = thrust::reduce(data_iter, data_iter + N, init, max_xy<Vec2>());
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec2>
GRACE_HOST void max_vec2(
    const Vec2* d_data,
    const size_t N,
    Vec2* maxs)
{
    max_vec2(thrust::device_ptr<const Vec2>(d_data), N, maxs);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec2Iter template over the const Vec2*
// when provided with a non-const Vec2*. This template is therefore needed to
// correctly handle non-const Vec2* pointers.
template <typename Vec2>
GRACE_HOST void max_vec2(
    Vec2* d_data,
    const size_t N,
    Vec2* maxs)
{
    max_vec2(thrust::device_ptr<const Vec2>(d_data), N, maxs);
}

template <typename Vec2>
GRACE_HOST void max_vec2(
    const thrust::device_vector<Vec2>& d_data,
    Vec2* maxs)
{
    max_vec2(d_data.begin(), d_data.size(), maxs);
}

template <typename Vec2>
GRACE_HOST void max_vec2(
    const thrust::host_vector<Vec2>& h_data,
    Vec2* maxs)
{
    max_vec2(h_data.begin(), h_data.size(), maxs);
}

// Vec3Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename Vec3Iter, typename Vec3>
GRACE_HOST void max_vec3(
    Vec3Iter data_iter,
    const size_t N,
    Vec3* maxs)
{
    // This may incurr a non-negligible overhead, but it is guaranteed to
    // produce a corrrect result for all Vec3-compatible types.
    const Vec3 init = data_iter[0];
    *maxs = thrust::reduce(data_iter, data_iter + N, init, max_xyz<Vec3>());
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec3>
GRACE_HOST void max_vec3(
    const Vec3* d_data,
    const size_t N,
    Vec3* maxs)
{
    max_vec3(thrust::device_ptr<const Vec3>(d_data), N, maxs);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec3Iter template over the const Vec3*
// when provided with a non-const Vec3*. This template is therefore needed to
// correctly handle non-const Vec3* pointers.
template <typename Vec3>
GRACE_HOST void max_vec3(
    Vec3* d_data,
    const size_t N,
    Vec3* maxs)
{
    max_vec3(thrust::device_ptr<const Vec3>(d_data), N, maxs);
}

template <typename Vec3>
GRACE_HOST void max_vec3(
    const thrust::device_vector<Vec3>& d_data,
    Vec3* maxs)
{
    max_vec3(d_data.begin(), d_data.size(), maxs);
}

template <typename Vec3>
GRACE_HOST void max_vec3(
    const thrust::host_vector<Vec3>& h_data,
    Vec3* maxs)
{
    max_vec3(h_data.begin(), h_data.size(), maxs);
}

// Vec4Iter should be a thrust iterator, thrust::device_ptr, or any custom
// iterator subject to the constraints described for TIter in grace::min_max().
template <typename Vec4Iter, typename Vec4>
GRACE_HOST void max_vec4(
    Vec4Iter data_iter,
    const size_t N,
    Vec4* maxs)
{
    // This may incurr a non-negligible overhead, but it is guaranteed to
    // produce a corrrect result for all Vec4-compatible types.
    const Vec4 init = data_iter[0];
    *maxs = thrust::reduce(data_iter, data_iter + N, init, max_xyzw<Vec4>());
}

// d_data must be a pointer to DEVICE memory.
template <typename Vec4>
GRACE_HOST void max_vec4(
    const Vec4* d_data,
    const size_t N,
    Vec4* maxs)
{
    max_vec4(thrust::device_ptr<const Vec4>(d_data), N, maxs);
}

// d_data must be a pointer to DEVICE memory.
// The compiler will chose the generic Vec4Iter template over the const Vec4*
// when provided with a non-const Vec4*. This template is therefore needed to
// correctly handle non-const Vec4* pointers.
template <typename Vec4>
GRACE_HOST void max_vec4(
    Vec4* d_data,
    const size_t N,
    Vec4* maxs)
{
    max_vec4(thrust::device_ptr<const Vec4>(d_data), N, maxs);
}

template <typename Vec4>
GRACE_HOST void max_vec4(
    const thrust::device_vector<Vec4>& d_data,
    Vec4* maxs)
{
    max_vec4(d_data.begin(), d_data.size(), maxs);
}

template <typename Vec4>
GRACE_HOST void max_vec4(
    const thrust::host_vector<Vec4>& h_data,
    Vec4* maxs)
{
    max_vec4(h_data.begin(), h_data.size(), maxs);
}

} // namespace grace
