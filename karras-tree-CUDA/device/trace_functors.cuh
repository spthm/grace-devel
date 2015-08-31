#include "../error.h"
#include "../types.h"
#include "util.cuh"

namespace grace
{

// RayData structs.

template<typename T>
struct RayData_simple
{
    T data;
};

template <typename T, typename Real>
struct RayData_sphere
{
    T data;
    Real b2, dist;
};


// 'Null' functors, when functionality is not required.
// Only makes sense for Init(), RayEntry() and RayExit().

class Init_null
{
public:
    GRACE_DEVICE void operator()(const UserSmemPtr<char>& sm_ptr)
    {
        return;
    }
};

class RayEntry_null
{
    template <typename RayData>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 const RayData& ray_data,
                                 const UserSmemPtr<char>& ptr)
    {
        return;
    }
};

typedef typename RayEntry_null RayExit_null;


// RayEntry functors.

template <typename RayData>
class RayEntry_simple
{
public:
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        ray_data.data = 0;
    }
};

template <RayData_sphere<int, typename Real>>
class RayEntry_individual
{
private:
    // MGPU's segmented routines force ray offsets to be ints.
    const int* const offsets;

public:
    RayEntry_sphere(const int* const ray_offsets) : offsets(ray_offsets) {}

    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData_sphere<int, Real>& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        ray_data.data = offsets[ray_idx];
    }
};


// RayExit functors.

// T can be deduced.
template <typename RayData, typename T>
class RayExit_simple
{
private:
    T* const output;

public:
    RayExit_simple(T* const output) : output(output) {}

    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 const RayData& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        output[ray_idx] = ray_data.data;
    }
};


// Init functors.

// Copying a contiguous set of data to SMEM.
// T can be deduced.
template <typename T>
class InitGlobalToSmem
{
private:
    const T* const global_ptr;
    const int count;

public:
    InitGlobalToSmem(const T* const addr, const int count) :
        global_ptr(addr), count(count) {}

    GRACE_DEVICE void operator()(const UserSmemPtr<char>& sm_ptr)
    {
        // We *must* cast from the default pointer-to-char to the data type we
        // wish to store in shared memory.
        UserSmemPtr<T> T_ptr = sm_ptr;

        for (int i = threadIdx.x; i < count; i += blockDim.x)
        {
            T_ptr[i] = global_ptr[i];
        }

        // __syncthreads() is called by the trace kernel.
    }
};


// Intersection functors.

// Discards impact parameter squared and distance to intersection.
template <typename RayData>
class Intersect_sphere_bool
{
public:
    template <typename Real4, typename RayData>
    GRACE_DEVICE bool operator()(const Ray& ray, const Real4& sphere,
                                 const RayData& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        typedef typename Real4ToRealMapper<Real4>::type Real;

        Real dummy_b2, dummy_dist;
        return sphere_hit(ray, sphere, &dummy_b2, &dummy_dist)
    }
};

// Stores impact parameter squared and distance to intersection.
template <RayData_sphere<typename T, typename Real> >
class Intersect_sphere_b2dist
{
public:
    template <typename Real4>
    GRACE_DEVICE bool operator()(const Ray& ray, const Real4& sphere,
                                 RayData_sphere<T, Real>& ray_data,
                                 const UserSmemPtr<char>& unused)
    {
        return sphere_hit(ray, sphere, &ray_data.b2, &ray_data.dist);
    }
};


// OnHit functors.

template <typename RayData>
class OnHit_increment
{
public:
    template <typename T>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData& ray_data,
                                 const prim_idx, const T& primitive,
                                 const UserSmemPtr<char>& sm_ptr)
    {
        ++ray_data.data;
    }
};

// Accumulating per-ray kernel integrals.
template <RayData_sphere<typename Real, typename Real> >
class OnHit_sphere_cumulative
{
private:
    const int N_table;

public:
    OnHit_sphere_cumulative(const int N_table) : N_table(N_table) {}

    template <typename Real4>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData_sphere<Real, Real>& ray_data,
                                 const prim_idx, const Real4& sphere,
                                 const UserSmemPtr<char>& sm_ptr)
    {
        GRACE_ASSERT(sizeof(sphere.x) == sizeof(Real));

        // For simplicity, we do not template the type of the kernel integral
        // lookup table; it is always required to be double.
        UserSmemPtr<double> Wk_lookup = sm_ptr;

        Real ir = 1.f / sphere.w;
        Real b = (N_table - 1) * (sqrt(ray_data.b2) * ir);
        Real integral = lerp(b, Wk_lookup, N_table);
        integral *= (ir * ir);

        GRACE_ASSERT(integral >= 0);
        GRACE_ASSERT(ray_data.dist >= 0);

        ray_data.data += integral;
    }
};

// Storing per-ray kernel integrals, sphere indices and ray-particle distances.
// IntegerIdx can be deduced.
template <RayData_sphere<typename T, typename Real>, typename IntegerIdx>
class OnHit_sphere_individual
{
private:
    IntegerIdx* const indices;
    Real* const integrals;
    Real* const distances;
    const int N_table;

public:
    OnHit_sphere_individual(IntegerIdx* const indices, Real* const integrals,
                            Real* const distances, const int N_table) :
        indices(indices), integrals(integrals), distances(distances),
        N_table(N_table) {}

    template <typename Real4>
    GRACE_DEVICE void operator()(const int ray_idx, const Ray& ray,
                                 RayData_sphere<T, Real>& ray_data,
                                 const sphere_idx, const Real4& sphere,
                                 const UserSmemPtr<char>& sm_ptr)
    {
        GRACE_ASSERT(sizeof(sphere.x) == sizeof(Real));

        // For simplicity, we do not template the type of the kernel integral
        // lookup table; it is always required to be double.
        UserSmemPtr<double> Wk_lookup = sm_ptr;

        Real ir = 1.f / sphere.w;
        Real b = (N_table - 1) * (sqrt(ray_data.b2) * ir);
        Real integral = lerp(b, Wk_lookup, N_table);
        integral *= (ir * ir);

        indices[ray_data.data] = node.x + i;
        integrals[ray_data.data] = integral;
        distances[ray_data.data] = ray_data.dist;

        GRACE_ASSERT(integral >= 0);
        GRACE_ASSERT(ray_data.dist >= 0);

        ++ray_data.data;
    }
};

} // namespace grace
