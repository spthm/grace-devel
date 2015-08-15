#pragma once

#include <assert.h>
// assert() is only supported for devices of compute capability 2.0 and higher.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#endif

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "../device/intrinsics.cuh"
#include "../error.h"
#include "../kernel_config.h"
#include "../nodes.h"
#include "../ray.h"
#include "../types.h"

#ifdef GRACE_NODES_TEX
#define FETCH_NODE(nodes, i) tex1Dfetch(nodes##_tex, i)
#else
#define FETCH_NODE(nodes, i) nodes[i]
#endif

#ifdef GRACE_SPHERES_TEX
#define FETCH_SPHERE(spheres, i) tex1Dfetch(spheres##_tex, i)
#else
#define FETCH_SPHERE(spheres, i) spheres[i]
#endif


namespace grace {

//-----------------------------------------------------------------------------
// Textures for tree access within trace kernels.
//-----------------------------------------------------------------------------

#ifdef GRACE_NODES_TEX
// float4 since it contains hierarchy (1 x int4) and AABB (3 x float4) data;
// easier to treat as float and reinterpret as int when necessary.
texture<float4, cudaTextureType1D, cudaReadModeElementType> nodes_tex;
texture<int4, cudaTextureType1D, cudaReadModeElementType> leaves_tex;
#endif

#ifdef GRACE_SPHERES_TEX
texture<float4, cudaTextureType1D, cudaReadModeElementType> spheres_tex;
#endif

//-----------------------------------------------------------------------------
// Helper functions for tracing kernels.
//-----------------------------------------------------------------------------

const int N_table = 51;

template <typename Float>
struct KernelIntegrals
{
    static Float table[N_table];

};

template <typename Float>
    Float KernelIntegrals<Float>::table[N_table] = {
    1.90986019771937, 1.90563449910964, 1.89304415940934, 1.87230928086763,
    1.84374947679902, 1.80776276033034, 1.76481079856299, 1.71540816859939,
    1.66011373131439, 1.59952322363667, 1.53426266082279, 1.46498233888091,
    1.39235130929287, 1.31705223652377, 1.23977618317103, 1.16121278415369,
    1.08201943664419, 1.00288866679720, 0.924475767210246, 0.847415371038733,
    0.772316688105931, 0.699736940377312, 0.630211918937167, 0.564194562399538,
    0.502076205853037, 0.444144023534733, 0.390518196140658, 0.341148855945766,
    0.295941946237307, 0.254782896476983, 0.217538645099225, 0.184059547649710,
    0.154181189781890, 0.127726122453554, 0.104505535066266,
    8.432088120445191E-002, 6.696547102921641E-002, 5.222604427168923E-002,
    3.988433820097490E-002, 2.971866601747601E-002, 2.150552303075515E-002,
    1.502124104014533E-002, 1.004371608622562E-002, 6.354242122978656E-003,
    3.739494884706115E-003, 1.993729589156428E-003, 9.212900163813992E-004,
    3.395908945333921E-004, 8.287326418242995E-005, 7.387919939044624E-006,
    0.000000000000000E+000
    };

template <typename Float4, typename Float>
GRACE_HOST_DEVICE bool sphere_hit(
    const Ray& ray,
    const Float4& sphere,
    Float& b2,
    Float& dot_p)
{
    float px = sphere.x - ray.ox;
    float py = sphere.y - ray.oy;
    float pz = sphere.z - ray.oz;

    // Already normalized.
    float rx = ray.dx;
    float ry = ray.dy;
    float rz = ray.dz;

    // Distance to intersection.
    dot_p = px*rx + py*ry + pz*rz;

    // Impact parameter.
    float bx = px - dot_p*rx;
    float by = py - dot_p*ry;
    float bz = pz - dot_p*rz;
    b2 = bx*bx + by*by + bz*bz;

    if (b2 >= sphere.w*sphere.w)
        return false;

    // If dot_p < 0, the ray origin must be inside the sphere for an
    // intersection. We treat this edge-case as a miss.
    if (dot_p < 0.0f)
        return false;

    // If dot_p > ray length, the ray terminus must be inside the sphere for
    // an intersection. We treat this edge-case as a miss.
    if (dot_p >= ray.length)
        return false;

    // Otherwise, assume we have a hit.  This counts the following partial
    // intersections as hits:
    //     i) Ray starts inside sphere, before point of closest approach.
    //    ii) Ray ends inside sphere, beyond point of closest approach.
    return true;
}

namespace gpu {

GRACE_DEVICE int AABBs_hit(
    const float3 invd, const float3 origin, const float len,
    const float4 AABB_L,
    const float4 AABB_R,
    const float4 AABB_LR)
{
    float bx_L = (AABB_L.x - origin.x) * invd.x;
    float tx_L = (AABB_L.y - origin.x) * invd.x;
    float by_L = (AABB_L.z - origin.y) * invd.y;
    float ty_L = (AABB_L.w - origin.y) * invd.y;
    float bz_L = (AABB_LR.x - origin.z) * invd.z;
    float tz_L = (AABB_LR.y - origin.z) * invd.z;

    float bx_R = (AABB_R.x - origin.x) * invd.x;
    float tx_R = (AABB_R.y - origin.x) * invd.x;
    float by_R = (AABB_R.z - origin.y) * invd.y;
    float ty_R = (AABB_R.w - origin.y) * invd.y;
    float bz_R = (AABB_LR.z - origin.z) * invd.z;
    float tz_R = (AABB_LR.w - origin.z) * invd.z;

    float tmin_L = maxf_vmaxf( fmin(bx_L, tx_L), fmin(by_L, ty_L),
                               maxf_vminf(bz_L, tz_L, 0) );
    float tmax_L = minf_vminf( fmax(bx_L, tx_L), fmax(by_L, ty_L),
                               minf_vmaxf(bz_L, tz_L, len) );
    float tmin_R = maxf_vmaxf( fmin(bx_R, tx_R), fmin(by_R, ty_R),
                               maxf_vminf(bz_R, tz_R, 0) );
    float tmax_R = minf_vminf( fmax(bx_R, tx_R), fmax(by_R, ty_R),
                               minf_vmaxf(bz_R, tz_R, len) );

    return (int)(tmax_L >= tmin_L) + 2*((int)(tmax_R >= tmin_R));
}

//-----------------------------------------------------------------------------
// CUDA tracing kernels.
//-----------------------------------------------------------------------------

template <typename Float4>
__global__ void trace_hitcounts_kernel(
    const Ray* rays,
    const size_t n_rays,
    int* hit_counts,
    const float4* nodes,
    size_t n_nodes,
    const int4* leaves,
    const int* root_index,
    const Float4* spheres,
    const size_t n_spheres,
    const int max_per_leaf)
{
    int tid = threadIdx.x % grace::WARP_SIZE; // ID of thread within warp.
    int wid = threadIdx.x / grace::WARP_SIZE; // ID of warp within block.
    int ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    // The index of the root node, where tracing begins.
    int root = *root_index;

    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t spheres_size = max_per_leaf * N_warps;

    extern __shared__ int smem[];
    // Offsets into the kernel's shared memory allocation must ensure correct
    // allignment!
    float4* sm_spheres = reinterpret_cast<float4*>(smem);
    int* sm_stacks = reinterpret_cast<int*>(&sm_spheres[spheres_size]);
    int* stack_ptr = sm_stacks + grace::STACK_SIZE * wid;

    // This is the exit sentinel. All threads in a ray packet (i.e. warp) write
    // to the same location to avoid any need for volatile declarations, or
    // warp-synchronous instructions (as far as the stack is concerned).
    *stack_ptr = -1;

    while (ray_index < n_rays)
    {
        int ray_hit_count = 0;

        Ray ray = rays[ray_index];

        float3 invd, origin;
        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        origin.x = ray.ox;
        origin.y = ray.oy;
        origin.z = ray.oz;

        // Push root to stack
        stack_ptr++;
        *stack_ptr = root;

        while (*stack_ptr >= 0)
        {
            // Nodes with an index > n_nodes are leaves. But, it is not safe to
            // compare signed (*stack_ptr) to unsigned (n_nodes) unless the
            // signed >= 0. This is also our stack-empty check.
            while (*stack_ptr < n_nodes && *stack_ptr >= 0)
            {
                assert(4*(*stack_ptr) + 3 < 4*n_nodes);

                // Pop stack.
                // If we immediately do a reinterpret_cast, the compiler states:
                // warning: taking the address of a temporary.
                float4 tmp = FETCH_NODE(nodes, 4*(*stack_ptr) + 0);
                int4 node = *reinterpret_cast<int4*>(&tmp);
                float4 AABB_L =  FETCH_NODE(nodes, 4*(*stack_ptr) + 1);
                float4 AABB_R =  FETCH_NODE(nodes, 4*(*stack_ptr) + 2);
                float4 AABB_LR = FETCH_NODE(nodes, 4*(*stack_ptr) + 3);
                stack_ptr--;

                // A left child can be nodes[0].
                // A right child cannot be nodes[0].
                assert(node.x >= 0);
                assert(node.y > 0);
                // Similarly for the last nodes, noting leaf indices are offset
                // by += n_nodes.
                assert(node.x < 2 * n_nodes);
                assert(node.y <= 2 * n_nodes);

                int lr_hit = AABBs_hit(invd, origin, ray.length,
                                       AABB_L, AABB_R, AABB_LR);

                // If any hit right child, push it to the stack.
                if (__any(lr_hit >= 2))
                {
                    stack_ptr++;
                    *stack_ptr = node.y;
                }
                // If any hit left child, push it to the stack.
                if (__any(lr_hit & 1u))
                {
                    stack_ptr++;
                    *stack_ptr = node.x;
                }

                assert(stack_ptr < sm_stacks + grace::STACK_SIZE * (wid + 1));

            }

            while (*stack_ptr >= n_nodes && *stack_ptr >= 0)
            {
                // Pop stack.
                int4 node = FETCH_NODE(leaves, (*stack_ptr)-n_nodes);
                assert(((*stack_ptr)-n_nodes) < n_nodes+1);
                stack_ptr--;
                assert(node.x >= 0);
                assert(node.y > 0);
                assert(node.x+node.y-1 < n_spheres);

                // Cache spheres in the leaf to shared memory. Coalesced.
                // For max_per_leaf == WARP_SIZE, it is quicker to read more
                // spheres than are in this leaf (which are not processed in the
                // following loop) than to have an if (tid < 32).
                // This additionally requires n_spheres be a multiple of 32.
                // sm_spheres[32*wid+tid] = FETCH_SPHERE(spheres, node.x+tid);
                for (int i=tid; i<node.y; i+=grace::WARP_SIZE)
                {
                    sm_spheres[max_per_leaf*wid+i] = FETCH_SPHERE(spheres,
                                                                  node.x+i);
                }
                // WARP-SYNCHRONOUS FIX:
                // __syncthreads();

                // Loop through spheres.
                for (int i=0; i<node.y; i++)
                {
                    // Unused.
                    float b2, dist;
                    if (sphere_hit(ray, sm_spheres[max_per_leaf*wid+i],
                                   b2, dist))
                    {
                        ray_hit_count++;
                    }
                }
            }

        }
        hit_counts[ray_index] = ray_hit_count;
        ray_index += blockDim.x * gridDim.x;
    }
}

template <typename Tout, typename Float4, typename Tin, typename Float>
__global__ void trace_property_kernel(
    const Ray* rays,
    const size_t n_rays,
    Tout* out_data,
    const float4* nodes,
    const size_t n_nodes,
    const int4* leaves,
    const int* root_index,
    const Float4* spheres,
    const int max_per_leaf,
    const Tin* p_data,
    const Float* g_b_integrals)
{
    int tid = threadIdx.x % grace::WARP_SIZE;
    int wid = threadIdx.x / grace::WARP_SIZE;
    int ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    int root = *root_index;

    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t spheres_size = max_per_leaf * N_warps;

    extern __shared__ int smem[];
    // Offsets into the kernel's shared memory allocation must ensure correct
    // allignment!
    float4* sm_spheres = reinterpret_cast<float4*>(smem);
    Float* b_integrals = reinterpret_cast<Float*>(&sm_spheres[spheres_size]);
    int* sm_stacks = reinterpret_cast<int*>(&b_integrals[grace::N_table]);
    int* stack_ptr = sm_stacks + grace::STACK_SIZE * wid;

    for (int i=threadIdx.x; i<N_table; i+=grace::TRACE_THREADS_PER_BLOCK)
    {
        b_integrals[i] = g_b_integrals[i];
    }
    __syncthreads();

    *stack_ptr = -1;

    while (ray_index < n_rays)
    {
        // Property to trace and accumulate.
        Tout out = 0;

        Ray ray = rays[ray_index];

        float3 invd, origin;
        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        origin.x = ray.ox;
        origin.y = ray.oy;
        origin.z = ray.oz;

        stack_ptr++;
        *stack_ptr = root;

        while (*stack_ptr >= 0)
        {
            while (*stack_ptr < n_nodes && *stack_ptr >= 0)
            {
                float4 tmp = FETCH_NODE(nodes, 4*(*stack_ptr) + 0);
                int4 node = *reinterpret_cast<int4*>(&tmp);
                float4 AABB_L =  FETCH_NODE(nodes, 4*(*stack_ptr) + 1);
                float4 AABB_R =  FETCH_NODE(nodes, 4*(*stack_ptr) + 2);
                float4 AABB_LR = FETCH_NODE(nodes, 4*(*stack_ptr) + 3);
                stack_ptr--;

                int lr_hit = AABBs_hit(invd, origin, ray.length,
                                       AABB_L, AABB_R, AABB_LR);

                if (__any(lr_hit >= 2))
                {
                    stack_ptr++;
                    *stack_ptr = node.y;
                }
                if (__any(lr_hit & 1u))
                {
                    stack_ptr++;
                    *stack_ptr = node.x;
                }
            }

            while (*stack_ptr >= n_nodes && *stack_ptr >= 0)
            {
                int4 node = FETCH_NODE(leaves, (*stack_ptr)-n_nodes);
                stack_ptr--;

                for (int i=tid; i<node.y; i+=grace::WARP_SIZE)
                {
                    sm_spheres[max_per_leaf*wid+i] = FETCH_SPHERE(spheres,
                                                                  node.x+i);
                }
                // WARP-SYNCHRONOUS FIX:
                // __syncthreads();

                for (int i=0; i<node.y; i++)
                {
                    int b_index;
                    float b, dist;
                    float4 sphere = sm_spheres[max_per_leaf*wid+i];
                    if (sphere_hit(ray, sphere, b, dist))
                    {
                        float ir = 1.f / sphere.w;
                        // sphere_hit returns |b*b|;
                        b = sqrtf(b);
                        // Normalize impact parameter to size of lookup table and
                        // interpolate.
                        b = (N_table-1) * (b * ir);
                        b_index = (int) b; // == floor(b)
                        if (b_index >= (N_table-1)) {
                            b = N_table-1;
                            b_index = N_table-2;
                        }
                        Float kernel_fac =
                            (b_integrals[b_index+1] - b_integrals[b_index])
                            * (b - b_index)
                            + b_integrals[b_index];
                        // Re-scale integral (since we used a normalized b).
                        kernel_fac *= (ir*ir);
                        assert(kernel_fac >= 0);
                        out += (Tout) (kernel_fac * p_data[node.x+i]);
                    }
                }
            }
        }
        out_data[ray_index] = out;
        ray_index += blockDim.x * gridDim.x;
    }
}

template <typename Tout, typename Float, typename Float4, typename Tin>
__global__ void trace_kernel(
    const Ray* rays,
    const size_t n_rays,
    Tout* out_data,
    unsigned int* hit_indices,
    Float* hit_distances,
    const int* ray_offsets,
    const float4* nodes,
    const size_t n_nodes,
    const int4* leaves,
    const int* root_index,
    const Float4* spheres,
    const int max_per_leaf,
    const Tin* p_data,
    const Float* g_b_integrals)
{
    int tid = threadIdx.x % grace::WARP_SIZE;
    int wid = threadIdx.x / grace::WARP_SIZE;
    int ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    int root = *root_index;

    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t spheres_size = max_per_leaf * N_warps;

    extern __shared__ int smem[];
    // Offsets into the kernel's shared memory allocation must ensure correct
    // allignment!
    float4* sm_spheres = reinterpret_cast<float4*>(smem);
    Float* b_integrals = reinterpret_cast<Float*>(&sm_spheres[spheres_size]);
    int* sm_stacks = reinterpret_cast<int*>(&b_integrals[grace::N_table]);
    int* stack_ptr = sm_stacks + grace::STACK_SIZE * wid;

    for (int i=threadIdx.x; i<N_table; i+=grace::TRACE_THREADS_PER_BLOCK)
    {
        b_integrals[i] = g_b_integrals[i];
    }
    __syncthreads();

    *stack_ptr = -1;

    while (ray_index < n_rays)
    {
        int out_index = ray_offsets[ray_index];

        Ray ray = rays[ray_index];

        float3 invd, origin;
        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        origin.x = ray.ox;
        origin.y = ray.oy;
        origin.z = ray.oz;

        stack_ptr++;
        *stack_ptr = root;

        while (*stack_ptr >= 0)
        {
            while (*stack_ptr < n_nodes && *stack_ptr >= 0)
            {
                float4 tmp = FETCH_NODE(nodes, 4*(*stack_ptr) + 0);
                int4 node = *reinterpret_cast<int4*>(&tmp);
                float4 AABB_L =  FETCH_NODE(nodes, 4*(*stack_ptr) + 1);
                float4 AABB_R =  FETCH_NODE(nodes, 4*(*stack_ptr) + 2);
                float4 AABB_LR = FETCH_NODE(nodes, 4*(*stack_ptr) + 3);
                stack_ptr--;

                int lr_hit = AABBs_hit(invd, origin, ray.length,
                                       AABB_L, AABB_R, AABB_LR);

                if (__any(lr_hit >= 2))
                {
                    stack_ptr++;
                    *stack_ptr = node.y;
                }
                if (__any(lr_hit & 1u))
                {
                    stack_ptr++;
                    *stack_ptr = node.x;
                }
            }

            while (*stack_ptr >= n_nodes && *stack_ptr >= 0)
            {
                int4 node = FETCH_NODE(leaves, (*stack_ptr)-n_nodes);
                stack_ptr--;

                for (int i=tid; i<node.y; i+=grace::WARP_SIZE)
                {
                    sm_spheres[max_per_leaf*wid+i] = FETCH_SPHERE(spheres,
                                                                  node.x+i);
                }
                // WARP-SYNCHRONOUS FIX:
                // __syncthreads();

                for (int i=0; i<node.y; i++)
                {
                    int b_index;
                    float b, dist;
                    float4 sphere = sm_spheres[max_per_leaf*wid+i];
                    if (sphere_hit(ray, sphere, b, dist))
                    {
                        float ir = 1.f / sphere.w;
                        b = sqrtf(b);
                        b = (N_table-1) * (b * ir);
                        b_index = (int) b;
                        if (b_index >= (N_table-1)) {
                            b = N_table-1;
                            b_index = N_table-2;
                        }
                        Float kernel_fac =
                            (b_integrals[b_index+1] - b_integrals[b_index])
                            * (b - b_index)
                            + b_integrals[b_index];
                        kernel_fac *= (ir*ir);
                        assert(kernel_fac >= 0);
                        assert(dist >= 0);
                        out_data[out_index] =
                            (Tout) (kernel_fac * p_data[node.x+i]);
                        hit_indices[out_index] = node.x+i;
                        hit_distances[out_index] = dist;
                        out_index++;
                    }
                }
            }
        }
        ray_index += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for tracing kernels
//-----------------------------------------------------------------------------

template <typename Float4>
GRACE_HOST void trace_hitcounts(
    const thrust::device_vector<Ray>& d_rays,
    thrust::device_vector<int>& d_hit_counts,
    const Tree& d_tree,
    const thrust::device_vector<Float4>& d_spheres)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_tree.leaves.size() - 1;

    int blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_rays + grace::TRACE_THREADS_PER_BLOCK - 1)
                             / grace::TRACE_THREADS_PER_BLOCK));

#if defined(GRACE_NODES_TEX) || defined(GRACE_SPHERES_TEX)
    cudaError_t cuerr;
#endif

#ifdef GRACE_NODES_TEX
    cuerr = cudaBindTexture(
        0, nodes_tex,
        reinterpret_cast<const float4*>(thrust::raw_pointer_cast(d_tree.nodes.data())),
        d_tree.nodes.size()*sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);

    cuerr = cudaBindTexture(
        0, leaves_tex, thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.leaves.size()*sizeof(int4));
    GRACE_CUDA_CHECK(cuerr);
#endif

#ifdef GRACE_SPHERES_TEX
    cuerr = cudaBindTexture(
        0, spheres_tex,
        thrust::raw_pointer_cast(d_spheres.data()),
        d_spheres.size()*sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);
#endif

    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t sm_size = sizeof(float4) * d_tree.max_per_leaf * N_warps
                           + sizeof(int) * grace::STACK_SIZE * N_warps;
    gpu::trace_hitcounts_kernel<<<blocks, grace::TRACE_THREADS_PER_BLOCK, sm_size>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        thrust::raw_pointer_cast(d_hit_counts.data()),
        reinterpret_cast<const float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.root_index_ptr,
        thrust::raw_pointer_cast(d_spheres.data()),
        d_spheres.size(),
        d_tree.max_per_leaf);
    GRACE_KERNEL_CHECK();

#ifdef GRACE_NODES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(nodes_tex));
    GRACE_CUDA_CHECK(cudaUnbindTexture(leaves_tex));
#endif
#ifdef GRACE_SPHERES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(spheres_tex));
#endif
}

template <typename Float, typename Tout, typename Float4, typename Tin>
GRACE_HOST void trace_property(
    const thrust::device_vector<Ray>& d_rays,
    thrust::device_vector<Tout>& d_out_data,
    const Tree& d_tree,
    const thrust::device_vector<Float4>& d_spheres,
    const thrust::device_vector<Tin>& d_in_data)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_tree.leaves.size() - 1;

    // TODO: Change it such that this is passed in, rather than instantiating
    // and copying it on each call to trace_property and trace.
    // Or make it static and initalize it in e.g. a grace_init function, that
    // could also determine kernel launch parameters.
    const KernelIntegrals<Float> lookup;
    const Float* p_table = &(lookup.table[0]);
    thrust::device_vector<Float> d_lookup(p_table, p_table + N_table);

    int blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_rays + grace::TRACE_THREADS_PER_BLOCK - 1)
                             / grace::TRACE_THREADS_PER_BLOCK));

#if defined(GRACE_NODES_TEX) || defined(GRACE_SPHERES_TEX)
    cudaError_t cuerr;
#endif

#ifdef GRACE_NODES_TEX
    cuerr = cudaBindTexture(
        0, nodes_tex,
        reinterpret_cast<const float4*>(thrust::raw_pointer_cast(d_tree.nodes.data())),
        d_tree.nodes.size()*sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);

    cuerr = cudaBindTexture(
        0, leaves_tex, thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.leaves.size()*sizeof(int4));
    GRACE_CUDA_CHECK(cuerr);
#endif

#ifdef GRACE_SPHERES_TEX
    cuerr = cudaBindTexture(
        0, spheres_tex,
        thrust::raw_pointer_cast(d_spheres.data()),
        d_spheres.size()*sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);
#endif

    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t sm_size = sizeof(float4) * d_tree.max_per_leaf * N_warps
                           + sizeof(Float) * grace::N_table
                           + sizeof(int) * grace::STACK_SIZE * N_warps;
    gpu::trace_property_kernel<<<blocks, grace::TRACE_THREADS_PER_BLOCK, sm_size>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        thrust::raw_pointer_cast(d_out_data.data()),
        reinterpret_cast<const float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.root_index_ptr,
        thrust::raw_pointer_cast(d_spheres.data()),
        d_tree.max_per_leaf,
        thrust::raw_pointer_cast(d_in_data.data()),
        thrust::raw_pointer_cast(d_lookup.data()));
    GRACE_KERNEL_CHECK();

#ifdef GRACE_NODES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(nodes_tex));
    GRACE_CUDA_CHECK(cudaUnbindTexture(leaves_tex));
#endif
#ifdef GRACE_SPHERES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(spheres_tex));
#endif
}

// TODO: Allow the user to supply correctly-sized hit-distance and output
//       arrays to this function, handling any memory issues therein themselves.
template <typename Float, typename Tout, typename Float4, typename Tin>
GRACE_HOST void trace(
    const thrust::device_vector<Ray>& d_rays,
    thrust::device_vector<Tout>& d_out_data,
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<unsigned int>& d_hit_indices,
    thrust::device_vector<Float>& d_hit_distances,
    const Tree& d_tree,
    const thrust::device_vector<Float4>& d_spheres,
    const thrust::device_vector<Tin>& d_in_data)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_tree.leaves.size() - 1;

    // Here, d_ray_offsets is actually per-ray *hit counts*.
    trace_hitcounts(d_rays, d_ray_offsets, d_tree, d_spheres);
    unsigned int last_ray_hits = d_ray_offsets[n_rays-1];

    // Allocate output array based on per-ray hit counts, and calculate
    // individual ray offsets into this array:
    //
    // hits = [3, 0, 4, 1]
    // exclusive_scan:
    //    => offsets = [0, 3, 3, 7]
    // total_hits = hits[3] + offsets[3] = 7 + 1 = 8
    thrust::exclusive_scan(d_ray_offsets.begin(), d_ray_offsets.end(),
                           d_ray_offsets.begin());
    unsigned int total_hits = d_ray_offsets[n_rays-1] + last_ray_hits;

    d_out_data.resize(total_hits);
    d_hit_indices.resize(total_hits);
    d_hit_distances.resize(total_hits);

    // TODO: Change it such that this is passed in, rather than instantiating
    // and copying it on each call to trace_property and trace.
    // Or initialize it as static, above, for both float and double.
    // Also then copy it into device memory?
    const KernelIntegrals<Float> lookup;
    const Float* p_table = &(lookup.table[0]);
    thrust::device_vector<Float> d_lookup(p_table, p_table + N_table);

    int blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_rays + grace::TRACE_THREADS_PER_BLOCK - 1)
                             / grace::TRACE_THREADS_PER_BLOCK));

#if defined(GRACE_NODES_TEX) || defined(GRACE_SPHERES_TEX)
    cudaError_t cuerr;
#endif

#ifdef GRACE_NODES_TEX
    cuerr = cudaBindTexture(
        0, nodes_tex,
        reinterpret_cast<const float4*>(thrust::raw_pointer_cast(d_tree.nodes.data())),
        d_tree.nodes.size()*sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);

    cuerr = cudaBindTexture(
        0, leaves_tex, thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.leaves.size()*sizeof(int4));
    GRACE_CUDA_CHECK(cuerr);
#endif

#ifdef GRACE_SPHERES_TEX
    cuerr = cudaBindTexture(
        0, spheres_tex,
        thrust::raw_pointer_cast(d_spheres.data()),
        d_spheres.size()*sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);
#endif

    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t sm_size = sizeof(float4) * d_tree.max_per_leaf * N_warps
                           + sizeof(Float) * grace::N_table
                           + sizeof(int) * grace::STACK_SIZE * N_warps;
    gpu::trace_kernel<<<blocks, grace::TRACE_THREADS_PER_BLOCK, sm_size>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        thrust::raw_pointer_cast(d_out_data.data()),
        thrust::raw_pointer_cast(d_hit_indices.data()),
        thrust::raw_pointer_cast(d_hit_distances.data()),
        thrust::raw_pointer_cast(d_ray_offsets.data()),
        reinterpret_cast<const float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.root_index_ptr,
        thrust::raw_pointer_cast(d_spheres.data()),
        d_tree.max_per_leaf,
        thrust::raw_pointer_cast(d_in_data.data()),
        thrust::raw_pointer_cast(d_lookup.data()));
    GRACE_KERNEL_CHECK();

#ifdef GRACE_NODES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(nodes_tex));
    GRACE_CUDA_CHECK(cudaUnbindTexture(leaves_tex));
#endif
#ifdef GRACE_SPHERES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(spheres_tex));
#endif
}

// TODO: Break this and trace() into multiple functions.
template <typename Float, typename Tout, typename Float4, typename Tin>
GRACE_HOST void trace_with_sentinels(
    const thrust::device_vector<Ray>& d_rays,
    thrust::device_vector<Tout>& d_out_data,
    const Tout out_sentinel,
    thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<unsigned int>& d_hit_indices,
    const unsigned int hit_sentinel,
    thrust::device_vector<Float>& d_hit_distances,
    const Float distance_sentinel,
    const Tree& d_tree,
    const thrust::device_vector<Float4>& d_spheres,
    const thrust::device_vector<Tin>& d_in_data)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_tree.leaves.size() - 1;

    // Here, d_ray_offsets is actually per-ray *hit counts*.
    trace_hitcounts(d_rays, d_ray_offsets, d_tree, d_spheres);
    unsigned int last_ray_hits = d_ray_offsets[n_rays-1];

    // Allocate output array based on per-ray hit counts, and calculate
    // individual ray offsets into this array:
    //
    // hits = [3, 0, 4, 1]
    // exclusive_scan:
    //     => offsets = [0, 3, 3, 7]
    thrust::exclusive_scan(d_ray_offsets.begin(), d_ray_offsets.end(),
                           d_ray_offsets.begin());
    size_t allocate_size = d_ray_offsets[n_rays-1] + last_ray_hits;
    // Increase the offsets such that they are correct if each ray segment in
    // the output array(s) ends with a dummy particle, or sentinel value,
    // marking the end of that ray's data.
    // transform:
    //     => offsets = [0, 4, 5, 10]
    allocate_size += n_rays; // For ray-end markers.
    {
    thrust::device_vector<unsigned int> d_offsets_inc(n_rays);
    thrust::sequence(d_offsets_inc.begin(), d_offsets_inc.end(), 0u);
    thrust::transform(d_ray_offsets.begin(), d_ray_offsets.end(),
                      d_offsets_inc.begin(),
                      d_ray_offsets.begin(),
                      thrust::plus<unsigned int>());
    } // So temporary d_offsets_inc is destroyed.

    // Initially, outputs should be populated with their sentinel/dummy values,
    // since these are not touched during tracing.
    d_out_data.resize(allocate_size, out_sentinel);
    d_hit_indices.resize(allocate_size, hit_sentinel);
    d_hit_distances.resize(allocate_size, distance_sentinel);

    // TODO: Change it such that this is passed in, rather than instantiating
    // and copying it on each call to trace_property and trace.
    // Or initialize it as static, above, for both float and double.
    // Also then copy it into device memory?
    const KernelIntegrals<Float> lookup;
    const Float* p_table = &(lookup.table[0]);
    thrust::device_vector<Float> d_lookup(p_table, p_table + N_table);

    int blocks = min(grace::MAX_BLOCKS,
                     (int) ((n_rays + grace::TRACE_THREADS_PER_BLOCK - 1)
                             / grace::TRACE_THREADS_PER_BLOCK));

#if defined(GRACE_NODES_TEX) || defined(GRACE_SPHERES_TEX)
    cudaError_t cuerr;
#endif

#ifdef GRACE_NODES_TEX
    cuerr = cudaBindTexture(
        0, nodes_tex,
        reinterpret_cast<const float4*>(thrust::raw_pointer_cast(d_tree.nodes.data())),
        d_tree.nodes.size()*sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);

    cuerr = cudaBindTexture(
        0, leaves_tex, thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.leaves.size()*sizeof(int4));
    GRACE_CUDA_CHECK(cuerr);
#endif

#ifdef GRACE_SPHERES_TEX
    cuerr = cudaBindTexture(
        0, spheres_tex,
        thrust::raw_pointer_cast(d_spheres.data()),
        d_spheres.size()*sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);
#endif

    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t sm_size = sizeof(float4) * d_tree.max_per_leaf * N_warps
                           + sizeof(Float) * grace::N_table
                           + sizeof(int) * grace::STACK_SIZE * N_warps;
    gpu::trace_kernel<<<blocks, grace::TRACE_THREADS_PER_BLOCK, sm_size>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        thrust::raw_pointer_cast(d_out_data.data()),
        thrust::raw_pointer_cast(d_hit_indices.data()),
        thrust::raw_pointer_cast(d_hit_distances.data()),
        thrust::raw_pointer_cast(d_ray_offsets.data()),
        reinterpret_cast<const float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.root_index_ptr,
        thrust::raw_pointer_cast(d_spheres.data()),
        d_tree.max_per_leaf,
        thrust::raw_pointer_cast(d_in_data.data()),
        thrust::raw_pointer_cast(d_lookup.data()));
    GRACE_KERNEL_CHECK();

#ifdef GRACE_NODES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(nodes_tex));
    GRACE_CUDA_CHECK(cudaUnbindTexture(leaves_tex));
#endif
#ifdef GRACE_SPHERES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(spheres_tex));
#endif
}

} // namespace grace
