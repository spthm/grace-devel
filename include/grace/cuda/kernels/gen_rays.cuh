#pragma once

// Due to a bug in the version of Thrust provided with CUDA 6, this must appear
// before #include <thrust/sort.h>
#include <curand_kernel.h>

#include "grace/cuda/kernel_config.h"
#include "grace/cuda/sort.cuh"
#include "grace/cuda/kernels/morton.cuh"

#include "grace/generic/morton.h"
#include "grace/generic/vecmath.h"
#include "grace/generic/functors/centroid.h"

#include "grace/error.h"
#include "grace/ray.h"
#include "grace/types.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace grace {

namespace detail {

////////////////////////////////////////////////////////////////////////////////
// Helper functions for ray generation
////////////////////////////////////////////////////////////////////////////////

// Returns a 30-bit morton key for the given ray, which must have its normalized
// direction vector set.
GRACE_HOST_DEVICE uinteger32 ray_dir_morton_key(const Ray ray)
{
    return morton_key((ray.dx + 1) / 2.f,
                      (ray.dy + 1) / 2.f,
                      (ray.dz + 1) / 2.f);
}

// Normalizes the provided (dx, dy, dz) direction vector and ets ray.dx/dy/dz.
GRACE_DEVICE float set_normalized_ray_direction(
    const float dx,
    const float dy,
    const float dz,
    Ray& ray)
{
    float invR;

    #if __CUDACC_VER_MAJOR__ >= 7
    invR = rnorm3d(dx, dy, dz);
    #else
    invR = rsqrt(dx*dx + dy*dy + dz*dz);
    #endif

    ray.dx = dx * invR;
    ray.dy = dy * invR;
    ray.dz = dz * invR;

    return invR;
}

// map f.k in [0, 1] to f'.k in [a.k, b.k] for each k = x, y, z component.
GRACE_HOST_DEVICE float3 zero_one_to_a_b(
    const float3 f,
    const float3 a,
    const float3 b)
{
    return make_float3(f.x * (b.x - a.x) + a.x,
                       f.y * (b.y - a.y) + a.y,
                       f.z * (b.z - a.z) + a.z);
}


////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////

__global__ void init_PRNG_kernel(
    curandState* const prng_states,
    const unsigned long long seed,
    const int N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Following the cuRAND documentation, each thread receives the same
        // seed value, no offset, and a *different* sequence value.
        // This should prevent any correlations if a single state is used to
        // generate multiple random numbers.
        curand_init(seed, tid, 0, &prng_states[tid]);
    }
}

/* N normally distributed values (mean 0, fixed variance) normalized
 * to one gives us a uniform distribution on the unit N-dimensional
 * hypersphere. See e.g. Wolfram "[Hyper]Sphere Point Picking" and
 * http://www.math.niu.edu/~rusin/known-math/96/sph.rand
 */
template <typename Real4, typename KeyType>
__global__ void gen_uniform_rays_kernel(
    const curandState* const prng_states,
    const Real4 ol, // ox, oy, oz, length.
    Ray* const rays,
    KeyType* const keys,
    const size_t N_rays)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state = prng_states[tid];

    while (tid < N_rays)
    {
        float dx, dy, dz;
        Ray ray;

        dx = curand_normal(&state);
        dy = curand_normal(&state);
        dz = curand_normal(&state);
        set_normalized_ray_direction(dx, dy, dz, ray);

        // morton_key requires *floats* in (0, 1) for 30-bit keys.
        keys[tid] = ray_dir_morton_key(ray);

        ray.ox = ol.x;
        ray.oy = ol.y;
        ray.oz = ol.z;
        ray.length = ol.w;

        rays[tid] = ray;

        tid += blockDim.x * gridDim.x;
    }
}

template <typename Real4, typename KeyType>
__global__ void gen_uniform_rays_single_octant_kernel(
    const curandState* const prng_states,
    const Real4 ol, // ox, oy, oz, length.
    Ray* const rays,
    KeyType* const keys,
    const size_t N_rays,
    const enum Octants octant)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state = prng_states[tid];

    int3 sign;
    sign.x = (octant & 0x4) ? 1 : -1;
    sign.y = (octant & 0x2) ? 1 : -1;
    sign.z = (octant & 0x1) ? 1 : -1;

    while (tid < N_rays)
    {
        float dx, dy, dz;
        Ray ray;

        dx = sign.x * fabsf(curand_normal(&state));
        dy = sign.y * fabsf(curand_normal(&state));
        dz = sign.z * fabsf(curand_normal(&state));
        set_normalized_ray_direction(dx, dy, dz, ray);

        // morton_key requires *floats* in (0, 1) for 30-bit keys.
        keys[tid] = ray_dir_morton_key(ray);

        ray.ox = ol.x;
        ray.oy = ol.y;
        ray.oz = ol.z;
        ray.length = ol.w;

        rays[tid] = ray;

        tid += blockDim.x * gridDim.x;
    }
}

template <RaySortType SortType, typename Real3, typename PointType,
          typename KeyType>
__global__ void one_to_many_rays_kernel(
    const Real3 o, // ox, oy, oz.
    const PointType* const points,
    Ray* const rays,
    KeyType* const keys,
    const size_t N_rays)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        float dx, dy, dz, R, invR;
        Ray ray;

        PointType point = points[tid];

        dx = point.x - o.x;
        dy = point.y - o.y;
        dz = point.z - o.z;
        invR = set_normalized_ray_direction(dx, dy, dz, ray);
        R = 1.0 / static_cast<double>(invR);

        if (SortType == DirectionSort) {
            keys[tid] = ray_dir_morton_key(ray);
        }

        ray.ox = o.x;
        ray.oy = o.y;
        ray.oz = o.z;
        ray.length = R;

        rays[tid] = ray;

        tid += blockDim.x * gridDim.x;
    }
}

template <typename Real, typename Real3>
__global__ void plane_parallel_random_rays_kernel(
    const curandState* const prng_states,
    const int width,
    const int height,
    const size_t n_rays,
    const Real3 base,
    const Real3 delta_w,
    const Real3 delta_h,
    const Real length,
    const Real3 normal,
    Ray* const rays)
{
    GRACE_ASSERT(grace::WARP_SIZE == warpSize);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state = prng_states[tid];

    for ( ; tid < n_rays; tid += blockDim.x * gridDim.x)
    {
        // Index of the current ray-grid cell.
        const int i = tid % width;
        const int j = tid / width;

        // The minimum (a) and maximum (b) bounds of the current cell along
        // delta_w and delta_h.
        const float3 a_w = make_float3(i * delta_w.x,
                                       i * delta_w.y,
                                       i * delta_w.z);
        const float3 b_w = make_float3((i + 1) * delta_w.x,
                                       (i + 1) * delta_w.y,
                                       (i + 1) * delta_w.z);

        const float3 a_h = make_float3(j * delta_h.x,
                                       j * delta_h.y,
                                       j * delta_h.z);
        const float3 b_h = make_float3((j + 1) * delta_h.x,
                                       (j + 1) * delta_h.y,
                                       (j + 1) * delta_h.z);

        // rand_w and rand_h are, respectively, the fraction of the current
        // cell's width and height at which to position the origin.
        // Note that to remain on the specified plane, these fractions must be
        // identical for x, y and z. That is, the ray's origin must be of the
        // form
        //   O = base + W * delta_w + H * delta_h
        const float rand_w = curand_uniform(&state);
        const float rand_h = curand_uniform(&state);
        const float3 rand_w3 = make_float3(rand_w, rand_w, rand_w);
        const float3 rand_h3 = make_float3(rand_h, rand_h, rand_h);

        const float3 w = zero_one_to_a_b(rand_w3, a_w, b_w);
        const float3 h = zero_one_to_a_b(rand_h3, a_h, b_h);

        Ray ray;
        ray.dx = normal.x;
        ray.dy = normal.y;
        ray.dz = normal.z;

        ray.ox = base.x + w.x + h.x;
        ray.oy = base.y + w.y + h.y;
        ray.oz = base.z + w.z + h.z;

        ray.length = length;

        rays[tid] = ray;
    }
}

template <typename Real, typename Real3>
__global__ void orthogonal_projection_rays_kernel(
    const int width,
    const int height,
    const size_t n_rays,
    const Real3 base_centre,
    const Real3 delta_w,
    const Real3 delta_h,
    const Real length,
    const Real3 normal,
    Ray* const rays)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < n_rays;
         tid += blockDim.x * gridDim.x)
    {
        const int i = tid % width;
        const int j = tid / width;

        Ray ray;
        ray.dx = normal.x;
        ray.dy = normal.y;
        ray.dz = normal.z;

        ray.ox = base_centre.x + i * delta_w.x + j * delta_h.x;
        ray.oy = base_centre.y + i * delta_w.y + j * delta_h.y;
        ray.oz = base_centre.z + i * delta_w.z + j * delta_h.z;

        ray.length = length;

        rays[tid] = ray;
    }
}

template <typename Real, typename Real3>
__global__ void pinhole_camera_rays_kernel(
    const int resolution_x,
    const int resolution_y,
    const size_t n_rays,
    const Real aspect_ratio,
    const Real3 camera_position,
    const Real3 u,
    const Real3 v,
    const Real3 n,
    const Real length,
    Ray* const rays)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < n_rays;
         tid += blockDim.x * gridDim.x)
    {
        const int i = tid % resolution_x;
        const int j = tid / resolution_x;

        // +0.5 to offset to pixel centres.
        float x = (2 * ((i + 0.5f) / resolution_x) - 1) * aspect_ratio;
        float y = 1 - 2 * ((j + 0.5f) / resolution_y);
        // Correct value for z is constant, and rolled into n.
        float z = 1.f;

        float3 dir;
        dir.x = x * v.x + y * u.x + z * n.x;
        dir.y = x * v.y + y * u.y + z * n.y;
        dir.z = x * v.z + y * u.z + z * n.z;
        dir = normalize3(dir);

        Ray ray;
        ray.dx = dir.x;
        ray.dy = dir.y;
        ray.dz = dir.z;

        ray.ox = camera_position.x;
        ray.oy = camera_position.y;
        ray.oz = camera_position.z;

        ray.length = length;

        rays[tid] = ray;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Kernel wrappers
////////////////////////////////////////////////////////////////////////////////

// We launch only enough blocks to ~fill the hardware, minimizing the number
// of threads which must call the expensive curand_init().
//
// Up to max_states will be generated, and the number of states will be an
// integer multiple of factor.
//
// If max_states is not itself an integer multiple of factor, then the
// maximum number of states which may be returned is the smallest number greater
// than max_states which is a multiple of factor.
//
// The number of states is returned in N_states.
//
// Reusing the same state to produce multiple, different random numbers should
// not lead to any (significant) correlation effects.
GRACE_HOST void init_PRNG(
    const int max_states,
    const int factor,
    const unsigned long long seed,
    curandState** d_prng_states,
    int* N_states)
{
    const int block_size = 128; // For init_PRNG_kernel
    int device_ID, N_SMs;
    cudaDeviceProp prop;
    GRACE_CUDA_CHECK(cudaGetDevice(&device_ID));
    GRACE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_ID));
    N_SMs = prop.multiProcessorCount;

    // Ideally we would have the smaller of:
    //     max_states
    //     3 * N_SMs * block_size,
    // but this must be rounded up to a multiple of factor.
    int N = std::min(3 * N_SMs * block_size, max_states);
    N = factor * (N + factor - 1) / factor;
    *N_states = N;

    const int num_blocks = (N + block_size - 1) / block_size;

    cudaError_t cuerr = cudaMalloc(
        (void**)d_prng_states,
        N * sizeof(curandState));
    GRACE_CUDA_CHECK(cuerr);

    // Initialize the P-RNG states.
    init_PRNG_kernel<<<num_blocks, block_size>>>(
        *d_prng_states,
        seed,
        N);
    GRACE_KERNEL_CHECK();
}

template <typename Real>
GRACE_HOST void uniform_random_rays(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const unsigned long long seed)
{
    const float4 origin = make_float4(ox, oy, oz, length);

    curandState* d_prng_states;
    int N_states;
    init_PRNG(N_rays, RAYS_THREADS_PER_BLOCK, seed, &d_prng_states, &N_states);

    thrust::device_vector<uinteger32> d_keys(N_rays);

    // init_PRNG guarantees N_states is a multiple of RAYS_THREADS_PER_BLOCK.
    const int num_blocks = N_states / RAYS_THREADS_PER_BLOCK;
    gen_uniform_rays_kernel<<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
        d_prng_states,
        origin,
        d_rays_ptr,
        thrust::raw_pointer_cast(d_keys.data()),
        N_rays);
    GRACE_KERNEL_CHECK();

    GRACE_CUDA_CHECK(cudaFree(d_prng_states));

    thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                        thrust::device_ptr<Ray>(d_rays_ptr));
}

template <typename Real>
GRACE_HOST void uniform_random_rays_single_octant(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const Real length,
    const enum Octants octant,
    const unsigned long long seed)
{
    const float4 origin = make_float4(ox, oy, oz, length);

    curandState* d_prng_states;
    int N_states;
    init_PRNG(N_rays, RAYS_THREADS_PER_BLOCK, seed, &d_prng_states, &N_states);

    thrust::device_vector<uinteger32> d_keys(N_rays);

    // init_PRNG guarantees N_states is a multiple of RAYS_THREADS_PER_BLOCK.
    const int num_blocks = N_states / RAYS_THREADS_PER_BLOCK;
    gen_uniform_rays_single_octant_kernel
        <<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
            d_prng_states,
            origin,
            d_rays_ptr,
            thrust::raw_pointer_cast(d_keys.data()),
            N_rays,
            octant);
    GRACE_KERNEL_CHECK();

    GRACE_CUDA_CHECK(cudaFree(d_prng_states));

    thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                        thrust::device_ptr<Ray>(d_rays_ptr));
}

// No sorting of rays (useful when particles are already spatially sorted).
template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays_nosort(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const PointType* const d_points_ptr)
{
    const float3 origin = make_float3(ox, oy, oz);

    const int num_blocks = min(grace::MAX_BLOCKS,
                               (int) ((N_rays + RAYS_THREADS_PER_BLOCK - 1)
                                       / RAYS_THREADS_PER_BLOCK));

    one_to_many_rays_kernel<NoSort><<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
        origin,
        d_points_ptr,
        d_rays_ptr,
        (uinteger32*)NULL,
        N_rays);
    GRACE_KERNEL_CHECK();
}

// No bounding box information for points; just sort rays based on their
// direction vectors.
template <typename Real, typename PointType>
GRACE_HOST void one_to_many_rays_dirsort(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const PointType* const d_points_ptr)
{
    const float3 origin = make_float3(ox, oy, oz);

    const int num_blocks = min(grace::MAX_BLOCKS,
                               (int) ((N_rays + RAYS_THREADS_PER_BLOCK - 1)
                                       / RAYS_THREADS_PER_BLOCK));

    thrust::device_vector<uinteger32> d_keys(N_rays);

    one_to_many_rays_kernel<DirectionSort>
        <<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
            origin,
            d_points_ptr,
            d_rays_ptr,
            thrust::raw_pointer_cast(d_keys.data()),
            N_rays);
    GRACE_KERNEL_CHECK();

    thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                        thrust::device_ptr<Ray>(d_rays_ptr));
}

// Have bounding box information for points; sort rays based on their end
// points.
template <typename Real, typename Real3, typename PointType>
GRACE_HOST void one_to_many_rays_endsort(
    Ray* const d_rays_ptr,
    const size_t N_rays,
    const Real ox,
    const Real oy,
    const Real oz,
    const PointType* const d_points_ptr,
    const Real3 AABB_bot,
    const Real3 AABB_top)
{
    const float3 origin = make_float3(ox, oy, oz);

    thrust::device_vector<uinteger32> d_keys(N_rays);
    uinteger32* d_keys_ptr = thrust::raw_pointer_cast(d_keys.data());

    const int num_blocks = min(grace::MAX_BLOCKS,
                               (int) ((N_rays + RAYS_THREADS_PER_BLOCK - 1)
                                       / RAYS_THREADS_PER_BLOCK));

    one_to_many_rays_kernel<EndPointSort>
        <<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
            origin,
            d_points_ptr,
            d_rays_ptr,
            (uinteger32*)NULL,
            N_rays);
    GRACE_KERNEL_CHECK();

    morton_keys(d_points_ptr, N_rays, AABB_bot, AABB_top, d_keys_ptr,
                CentroidSphere());

    thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                        thrust::device_ptr<Ray>(d_rays_ptr));
}

template <typename Real, typename Real3>
GRACE_HOST void plane_parallel_random_rays(
    Ray* const d_rays_ptr,
    const int width,
    const int height,
    const Real3 base,
    const Real3 w,
    const Real3 h,
    const Real length,
    const unsigned long long seed)
{
    const size_t N_rays = static_cast<size_t>(width) * height;

    curandState* d_prng_states;
    int N_states;
    init_PRNG(N_rays, RAYS_THREADS_PER_BLOCK, seed, &d_prng_states, &N_states);

    Real3 delta_w, delta_h, direction;

    delta_w.x = w.x / width;
    delta_w.y = w.y / width;
    delta_w.z = w.z / width;

    delta_h.x = h.x / height;
    delta_h.y = h.y / height;
    delta_h.z = h.z / height;

    direction = normalize3(cross(w, h));

    // init_PRNG guarantees N_states is a multiple of RAYS_THREADS_PER_BLOCK.
    const int num_blocks = N_states / RAYS_THREADS_PER_BLOCK;
    plane_parallel_random_rays_kernel<<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
        d_prng_states,
        width,
        height,
        N_rays,
        base,
        delta_w,
        delta_h,
        length,
        direction,
        d_rays_ptr);
    GRACE_KERNEL_CHECK();

    GRACE_CUDA_CHECK(cudaFree(d_prng_states));
}

template <typename Real, typename Real3>
GRACE_HOST void orthogonal_projection_rays(
    Ray* const d_rays_ptr,
    const int width,
    const int height,
    const Real3 base,
    const Real3 w,
    const Real3 h,
    const Real length)
{
    const size_t N_rays = static_cast<size_t>(width) * height;

    Real3 delta_w, delta_h, base_centre, direction;

    delta_w.x = w.x / width;
    delta_w.y = w.y / width;
    delta_w.z = w.z / width;

    delta_h.x = h.x / height;
    delta_h.y = h.y / height;
    delta_h.z = h.z / height;

    base_centre.x = base.x + (delta_w.x + delta_h.x) / 2.;
    base_centre.y = base.y + (delta_w.y + delta_h.y) / 2.;
    base_centre.z = base.z + (delta_w.z + delta_h.z) / 2.;

    direction = normalize3(cross(w, h));

    const int num_blocks = min(grace::MAX_BLOCKS,
                               (int) ((N_rays + RAYS_THREADS_PER_BLOCK - 1)
                                       / RAYS_THREADS_PER_BLOCK));
    orthogonal_projection_rays_kernel<<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
        width,
        height,
        N_rays,
        base_centre,
        delta_w,
        delta_h,
        length,
        direction,
        d_rays_ptr);
    GRACE_KERNEL_CHECK();
}

template <typename Real, typename Real3>
GRACE_HOST void pinhole_camera_rays(
    Ray* const d_rays_ptr,
    const int resolution_x,
    const int resolution_y,
    const Real3 camera_position,
    const Real3 look_at,
    const Real3 view_up,
    const Real FOVy,
    const Real length)
{
    // The kernel generates the points at which each ray intersects the image
    // plane, but in the camera's co-ordinate system: the camera is centred at
    // the origin, facing the -z direction, with +x rightward and +y upward.
    // These points extend over (-aspect_ratio, aspect_ratio) along the x axis,
    // (1, -1) along the y axis, and are a distance -1 / tan(FOVy/2) from the
    // origin on the z axis, in a right-handed system.
    // Each ray's direction vector can then be computed, in the same
    // co-ordinate system, from each origin-to-point vector.
    // These directions are transformed to the world co-ordinate system using
    // a new basis, v, u, n, which is a left-handed system.

    const size_t N_rays = (size_t)resolution_x * resolution_y;
    const Real aspect_ratio = (Real)resolution_x / resolution_y;

    // Construct u, v, n basis of image plane. +u is upward in plane, +v is
    // rightward in plane, and +n is into plane. Again, assume a right-handed
    // system.

    Real3 L;
    L.x = look_at.x - camera_position.x;
    L.y = look_at.y - camera_position.y;
    L.z = look_at.z - camera_position.z;

    Real3 v = normalize3(cross(L, view_up));
    Real3 u = normalize3(cross(v, L));
    Real3 n = normalize3(L);

    // Do this once here rather than by each thread in the kernel.
    // In the camera's co-ordinate system, the image plane is at
    // -1 / tan(FOVy/2), but we're swapping from a RH to a LH system.
    Real n_prefactor = 1. / std::tan(FOVy / 2.);
    n.x *= n_prefactor;
    n.y *= n_prefactor;
    n.z *= n_prefactor;

    const int num_blocks = min(grace::MAX_BLOCKS,
                               (int) ((N_rays + RAYS_THREADS_PER_BLOCK - 1)
                                       / RAYS_THREADS_PER_BLOCK));
    pinhole_camera_rays_kernel<<<num_blocks, RAYS_THREADS_PER_BLOCK>>>(
        resolution_x,
        resolution_y,
        N_rays,
        aspect_ratio,
        camera_position,
        u,
        v,
        n,
        length,
        d_rays_ptr);
    GRACE_KERNEL_CHECK();
}

} // namespace detail

} // namespace grace
