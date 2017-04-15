#pragma once

#include "grace/cuda/detail/device/intersect.cuh"
#include "grace/cuda/detail/kernel_config.h"

#include "grace/cuda/util/texref_iter.cuh"

#include "grace/generic/boundedptr.h"
#include "grace/generic/meta.h"

#include "grace/cuda/nodes.h"

#include "grace/error.h"
#include "grace/ray.h"
#include "grace/types.h"

#include <thrust/device_vector.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

#define FETCH_NODE(nodes, i) tex1Dfetch(nodes##_tex, i)


namespace grace {

namespace gpu {

template <typename Primitive, typename RayData, typename Intersection,
          typename OnHit>
GRACE_DEVICE
void leaf_intersector(
    const int4& leaf, const Primitive* const leaf_prims, const Ray& ray,
    int ray_index, RayData& ray_data, Intersection intersect, OnHit on_hit,
    const BoundedPtr<char>& sm_ptr_user)
{
    const int lane = threadIdx.x % grace::WARP_SIZE;

    for (int i = 0; i < leaf.y; ++i)
    {
        const Primitive prim = leaf_prims[i];
        if (intersect(ray, prim, ray_data, lane, sm_ptr_user))
        {
            on_hit(ray_index, ray, ray_data, leaf.x + i, prim, lane,
                   sm_ptr_user);
        }
    }
}

template <typename Primitive, typename RayData, typename Intersection,
          typename OnHit>
GRACE_DEVICE
void leaf_intersector_rayloop_sm20(
    const int4& leaf, const Primitive* const leaf_prims, const Ray* rays,
    const int first_ray_index, RayData* rays_data, Intersection intersect,
    OnHit on_hit, const BoundedPtr<char>& sm_ptr_user)
{
    const int lane = threadIdx.x % grace::WARP_SIZE;

    for (int i = lane; i < leaf.y; i += grace::WARP_SIZE)
    {
        const Primitive prim = leaf_prims[i];

        for (int j = 0; j < grace::WARP_SIZE; ++j)
        {
            const Ray ray = rays[j];
            RayData ray_data = rays_data[j];

            bool hit = intersect(ray, prim, ray_data, lane, sm_ptr_user);
            if (hit)
            {
                on_hit(first_ray_index + j, ray, ray_data, leaf.x + i, prim,
                       lane, sm_ptr_user);
            }

            // There is an ambiguity for which ray data we update to.
            // We choose the highest-valued lane's version.
            int high_lane = grace::WARP_SIZE - __clz(__ballot(hit)) - 1;
            GRACE_ASSERT(high_lane >= 0);
            if (lane == high_lane)
            {
                rays_data[j] = ray_data;
            }
        }
    }
}

template <typename Primitive, typename RayData, typename Intersection,
          typename OnHit>
GRACE_DEVICE
void leaf_intersector_rayloop_sm30(
    const int4& leaf, const Primitive* const leaf_prims, const Ray& ray,
    const int first_ray_index, RayData& ray_data, Intersection intersect,
    OnHit on_hit, const BoundedPtr<char>& sm_ptr_user)
{
    const int lane = threadIdx.x % grace::WARP_SIZE;

    // All lanes have to hit the shfl_idx().
    const int n = (leaf.y + grace::WARP_SIZE - 1) / grace::WARP_SIZE;
    for (int k = 0; k < n; ++k)
    {
        const int i = lane + k * grace::WARP_SIZE;
        Primitive prim;
        if (i < leaf.y) prim = leaf_prims[i];

        for (int j = 0; j < grace::WARP_SIZE; ++j)
        {
            const Ray ray_j = shfl_idx(ray, j);
            RayData ray_data_j = shfl_idx(ray_data, j);

            bool hit = false;
            if (i < leaf.y)
            {
                hit = intersect(ray_j, prim, ray_data_j, lane, sm_ptr_user);
                if (hit)
                {
                    on_hit(first_ray_index + j, ray_j, ray_data_j, leaf.x + i,
                           prim, lane, sm_ptr_user);
                }
            }

            // There is an ambiguity for which ray data we update to.
            // We choose the highest-valued lane's version.
            int high_lane = grace::WARP_SIZE - __clz(__ballot(hit)) - 1;
            GRACE_ASSERT(high_lane >= 0);
            // All lanes must take part.
            ray_data_j = shfl_idx(ray_data_j, high_lane);
            if (lane == j)
            {
                ray_data = ray_data_j;
            }
        }
    }
}

template <typename T, typename U>
GRACE_HOST_DEVICE T* align_ptr(U* ptr)
{
    char* ptr_c = (char*)ptr;
    int rem = (uintptr_t)ptr_c % GRACE_ALIGNOF(T);
    if (rem != 0) {
        ptr_c += GRACE_ALIGNOF(T) - rem;
    }
    return reinterpret_cast<T*>(ptr_c);
}

//-----------------------------------------------------------------------------
// Textures for tree access within trace kernels.
//-----------------------------------------------------------------------------

// float4 since it contains hierarchy (1 x int4) and AABB (3 x float4) data;
// easier to treat as float and reinterpret as int when necessary.
texture<float4, cudaTextureType1D, cudaReadModeElementType> nodes_tex;
texture<int4, cudaTextureType1D, cudaReadModeElementType> leaves_tex;

//-----------------------------------------------------------------------------
// CUDA tracing kernel.
//-----------------------------------------------------------------------------

// RayIter _may_ be a const Ray* or const_iterator<Ray>.
// PrimitiveIter _may_ be a const TPrimitive* or const_iterator<TPrimitive>
template <typename RayData,
          LeafTraversal::E LTConfig,
          typename RayIter,
          typename PrimitiveIter,
          typename Init,
          typename Intersection, typename OnHit,
          typename OnRayEntry, typename OnRayExit>
__global__ void trace_kernel(
    RayIter rays,
    const size_t n_rays,
    const float4* nodes,
    const size_t n_nodes,
    const int4* leaves,
    const int* root_index,
    PrimitiveIter primitives,
    const size_t n_primitives,
    const int max_per_leaf,
    const size_t user_smem_bytes, // User's SMEM allocation, in bytes.
    Init init,              // pre-traversal functor
    Intersection intersect, // ray-primitive intersection test functor
    OnHit on_hit,           // ray-primitive intersection processing functor
    OnRayEntry ray_entry,   // ray-traversal entry functor
    OnRayExit ray_exit)     // ray-traversal exit functor
{
    typedef typename std::iterator_traits<RayIter>::value_type TRay;
    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;
    GRACE_STATIC_ASSERT( (are_same<TRay, grace::Ray>::result), "Ray type must be grace::Ray");

    const int lane = threadIdx.x % grace::WARP_SIZE;
    const int wid  = threadIdx.x / grace::WARP_SIZE;
    int ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    // The tree's root index can be anywhere in ALBVH.
    const int root = *root_index;

    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t prims_smem_count = max_per_leaf * N_warps;

    extern __shared__ char smem_trace[];
    const BoundedPtr<char> sm_iter_usr(smem_trace, user_smem_bytes);
    init(sm_iter_usr);
    __syncthreads();


    // Shared memory accesses must ensure correct alignment relative to the
    // access size.
    TPrimitive* sm_prims = align_ptr<TPrimitive>(smem_trace + user_smem_bytes);

    int* sm_stacks;
#if __CUDA_ARCH__ < 300
    Ray* sm_rays;
    RayData* sm_rays_data;

    if (LTConfig == LeafTraversal::ParallelPrimitives)
    {
        sm_rays = align_ptr<Ray>(sm_prims + prims_smem_count);
        sm_rays_data = align_ptr<RayData>(sm_rays + grace::WARP_SIZE * N_warps);
        sm_stacks = align_ptr<int>(sm_rays_data + grace::WARP_SIZE * N_warps);
    }
    else
    {
        sm_stacks = align_ptr<int>(sm_prims + prims_smem_count);
    }
#else
    sm_stacks = align_ptr<int>(sm_prims + prims_smem_count);
#endif

    // This warp's offset.
    int* stack_ptr = sm_stacks + grace::STACK_SIZE * wid;
    sm_prims += max_per_leaf * wid;
#if __CUDA_ARCH__ < 300
    if (LTConfig == LeafTraversal::ParallelPrimitives) {
        sm_rays += grace::WARP_SIZE * wid;
        sm_rays_data += grace::WARP_SIZE * wid;
    }
#endif

    // This is the exit sentinel. All threads in a ray packet (i.e. warp) write
    // to the same location to avoid any need for volatile declarations, or
    // warp-synchronous instructions (as far as the stack is concerned).
    *stack_ptr = -1;

    while (ray_index < n_rays)
    {
        // Ray must not be modified by user.
        const Ray ray = rays[ray_index];
        RayData ray_data = {};

    #if __CUDA_ARCH__ < 300
        if (LTConfig == LeafTraversal::ParallelPrimitives)
        {
            sm_rays[lane] = ray;
            sm_rays_data[lane] = ray_data;
        }
    #endif

        ray_entry(ray_index, ray, ray_data, sm_iter_usr);

        float3 invd, origin;
        invd.x = 1.f / ray.dx;
        invd.y = 1.f / ray.dy;
        invd.z = 1.f / ray.dz;
        origin.x = ray.ox;
        origin.y = ray.oy;
        origin.z = ray.oz;

        // Push root to stack.
        stack_ptr++;
        *stack_ptr = root;

        while (*stack_ptr >= 0)
        {
            // Nodes with an index > n_nodes are leaves. But, it is not safe to
            // compare signed (*stack_ptr) to unsigned (n_nodes) unless the
            // signed >= 0. This is also our stack-empty check.
            while (*stack_ptr < n_nodes && *stack_ptr >= 0)
            {
                GRACE_ASSERT(4 * (*stack_ptr) + 3 < 4 * n_nodes);

                // Pop stack.
                // If we immediately do a reinterpret_cast, the compiler states:
                // warning: taking the address of a temporary.
                float4 tmp = FETCH_NODE(nodes, 4*(*stack_ptr) + 0);
                int4 node = *reinterpret_cast<int4*>(&tmp);
                float4 AABB_L =  FETCH_NODE(nodes, 4*(*stack_ptr) + 1);
                float4 AABB_R =  FETCH_NODE(nodes, 4*(*stack_ptr) + 2);
                float4 AABB_LR = FETCH_NODE(nodes, 4*(*stack_ptr) + 3);
                stack_ptr--;

                GRACE_ASSERT(node.x >= 0);
                GRACE_ASSERT(node.y > 0);
                // Recall that leaf indices are offset by += n_nodes.
                GRACE_ASSERT(node.x < 2 * n_nodes);
                GRACE_ASSERT(node.y <= 2 * n_nodes);

                int lr_hit = AABBs_hit(invd, origin, ray.start, ray.end,
                                       AABB_L, AABB_R, AABB_LR);

                if (__any(lr_hit & 1u))
                {
                    stack_ptr++;
                    *stack_ptr = node.y;
                }
                if (__any(lr_hit >= 2))
                {
                    stack_ptr++;
                    *stack_ptr = node.x;
                }

                // FIXME: Produces compile-time warning.
                // See http://stackoverflow.com/questions/1712713/
                GRACE_ASSERT(stack_ptr < sm_stacks + grace::STACK_SIZE * (wid + 1), trace_stack_overflow);
            }

            while (*stack_ptr >= n_nodes && *stack_ptr >= 0)
            {
                // Pop stack.
                int4 node = FETCH_NODE(leaves, (*stack_ptr)-n_nodes);
                GRACE_ASSERT(((*stack_ptr) - n_nodes) < n_nodes + 1);
                stack_ptr--;

                GRACE_ASSERT(node.x >= 0);
                GRACE_ASSERT(node.y > 0);
                GRACE_ASSERT(node.x + node.y - 1 < n_primitives);

                for (int i = lane; i < node.y; i += grace::WARP_SIZE)
                {
                    sm_prims[i] = primitives[node.x + i];
                }

                if (LTConfig == LeafTraversal::ParallelRays)
                {
                    leaf_intersector(node, sm_prims, ray, ray_index, ray_data,
                                     intersect, on_hit, sm_iter_usr);
                }
                else if (LTConfig == LeafTraversal::ParallelPrimitives)
                {
                    int first_ray_index = ray_index - lane;
                #if __CUDA_ARCH__ < 300
                    leaf_intersector_rayloop_sm20(node,
                                                  sm_prims,
                                                  sm_rays,
                                                  first_ray_index,
                                                  sm_rays_data,
                                                  intersect,
                                                  on_hit,
                                                  sm_iter_usr);
                #else
                    leaf_intersector_rayloop_sm30(node,
                                                  sm_prims,
                                                  ray,
                                                  first_ray_index,
                                                  ray_data,
                                                  intersect,
                                                  on_hit,
                                                  sm_iter_usr);
                #endif
                }
                else
                {
                    GRACE_ASSERT(false);
                }
            }
        }
        ray_exit(ray_index, ray, ray_data, sm_iter_usr);
        ray_index += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrappers for trace kernel.
//-----------------------------------------------------------------------------

// RayIter _may_ be a const Ray* or const_iterator<Ray>.
// PrimitiveIter _may_ be a const TPrimitive* or const_iterator<TPrimitive>
// Both must be dereferencable on the device.
template <typename RayData,
          LeafTraversal::E LTConfig,
          typename RayIter,
          typename PrimitiveIter,
          typename Init,
          typename Intersection, typename OnHit,
          typename OnRayEntry, typename OnRayExit>
GRACE_HOST void trace(
    RayIter d_rays_iter,
    const size_t N_rays,
    PrimitiveIter d_prims_iter,
    const size_t N_primitives,
    const Tree& d_tree,
    const size_t user_smem_bytes,
    Init init,
    Intersection intersect,
    OnHit on_hit,
    OnRayEntry ray_entry,
    OnRayExit ray_exit)
{
    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;

    size_t N_nodes = d_tree.leaves.size() - 1;

    if (N_rays % grace::WARP_SIZE != 0) {
        std::stringstream msg_stream;
        msg_stream << "Number of rays must be a multiple of the warp size ("
                   << grace::WARP_SIZE << ").";
        const std::string msg = msg_stream.str();

        throw std::invalid_argument(msg);
    }

    cudaError_t cuerr;

    cuerr = cudaBindTexture(
        0, gpu::nodes_tex,
        reinterpret_cast<const float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        d_tree.nodes.size() * sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);

    cuerr = cudaBindTexture(
        0, gpu::leaves_tex, thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.leaves.size() * sizeof(int4));
    GRACE_CUDA_CHECK(cuerr);

    const int NT = grace::TRACE_THREADS_PER_BLOCK;
    const int blocks = std::min((int)((N_rays + NT - 1) / NT),
                                grace::MAX_BLOCKS);
    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t sm_size = sizeof(TPrimitive) * d_tree.max_per_leaf * N_warps
                           + sizeof(TPrimitive) - 1 // For alignment correction.
                           + sizeof(int) * grace::STACK_SIZE * N_warps
                           + user_smem_bytes;

    gpu::trace_kernel<RayData, LTConfig><<<blocks, NT, sm_size>>>(
        d_rays_iter,
        N_rays,
        reinterpret_cast<const float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        N_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.root_index_ptr,
        d_prims_iter,
        N_primitives,
        d_tree.max_per_leaf,
        user_smem_bytes,
        init,
        intersect,
        on_hit,
        ray_entry,
        ray_exit);
    GRACE_KERNEL_CHECK();

    GRACE_CUDA_CHECK(cudaUnbindTexture(gpu::nodes_tex));
    GRACE_CUDA_CHECK(cudaUnbindTexture(gpu::leaves_tex));
}

template <typename RayData,
          LeafTraversal::E LTConfig,
          typename TPrimitive,
          typename Init,
          typename Intersection, typename OnHit,
          typename OnRayEntry, typename OnRayExit>
GRACE_HOST void trace(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<TPrimitive>& d_primitives,
    const Tree& d_tree,
    const size_t user_smem_bytes,
    Init init,
    Intersection intersect,
    OnHit on_hit,
    OnRayEntry ray_entry,
    OnRayExit ray_exit)
{
    const Ray* d_rays_iter = thrust::raw_pointer_cast(d_rays.data());
    const TPrimitive* d_prims_iter = thrust::raw_pointer_cast(d_primitives.data());

    trace<RayData, LTConfig>(d_rays_iter, d_rays.size(), d_prims_iter,
                             d_primitives.size(), d_tree, user_smem_bytes, init,
                             intersect, on_hit, ray_entry, ray_exit);
}

// Reads the primitives through the texture cache.
// RayIter _may_ be a const Ray* or const_iterator<Ray>. It must be
// dereferencable on the device.
template <typename RayData,
          LeafTraversal::E LTConfig,
          typename RayIter,
          typename TPrimitive,
          typename Init,
          typename Intersection, typename OnHit,
          typename OnRayEntry, typename OnRayExit>
GRACE_HOST void trace_texref(
    RayIter d_rays_iter,
    const size_t N_rays,
    const TPrimitive* d_primitives,
    const size_t N_primitives,
    const Tree& d_tree,
    const size_t user_smem_bytes,
    Init init,
    Intersection intersect,
    OnHit on_hit,
    OnRayEntry ray_entry,
    OnRayExit ray_exit)
{
    TexRefIter<TPrimitive, PRIMITIVE_TEX_UID> prims_iter;

    cudaError_t cuerr
        = prims_iter.bind(d_primitives, N_primitives * sizeof(TPrimitive));
    GRACE_CUDA_CHECK(cuerr);

    trace<RayData, LTConfig>(d_rays_iter, N_rays, prims_iter, N_primitives,
                             d_tree, user_smem_bytes, init, intersect, on_hit,
                             ray_entry, ray_exit);

    GRACE_CUDA_CHECK(prims_iter.unbind());
}

template <typename RayData,
          LeafTraversal::E LTConfig,
          typename TPrimitive,
          typename Init,
          typename Intersection, typename OnHit,
          typename OnRayEntry, typename OnRayExit>
GRACE_HOST void trace_texref(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<TPrimitive>& d_primitives,
    const Tree& d_tree,
    const size_t user_smem_bytes,
    Init init,
    Intersection intersect,
    OnHit on_hit,
    OnRayEntry ray_entry,
    OnRayExit ray_exit)
{
    const TPrimitive* prims_ptr = thrust::raw_pointer_cast(d_primitives.data());
    const Ray* rays_ptr = thrust::raw_pointer_cast(d_rays.data());

    trace_texref<RayData, LTConfig>(rays_ptr, d_rays.size(), prims_ptr,
                                    d_primitives.size(), d_tree,
                                    user_smem_bytes, init, intersect, on_hit,
                                    ray_entry, ray_exit);
}

} // namespace grace
