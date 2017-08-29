#pragma once

#include "grace/cuda/detail/device/intersect.cuh"
#include "grace/cuda/detail/kernel_config.h"

#include "grace/cuda/util/texref_iter.cuh"

#include "grace/generic/boundedptr.h"
#include "grace/generic/meta.h"

#include "grace/cuda/bvh.cuh"
#include "grace/cuda/error.cuh"

#include "grace/detail/assert.h"

#include "grace/config.h"
#include "grace/ray.h"
#include "grace/types.h"

#include <thrust/device_vector.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>


namespace grace {

namespace detail {

template <typename Primitive, typename RayData, typename Intersection,
          typename OnHit>
GRACE_DEVICE
void leaf_intersector(
    const CudaBvhLeaf& leaf, const Primitive* const leaf_prims, const Ray& ray,
    int ray_index, RayData& ray_data, Intersection intersect, OnHit on_hit,
    const BoundedPtr<char>& sm_ptr_user)
{
    const int lane = threadIdx.x % grace::WARP_SIZE;

    for (int i = 0; i < leaf.size(); ++i)
    {
        const Primitive prim = leaf_prims[i];
        if (intersect(ray, prim, ray_data, lane, sm_ptr_user))
        {
            on_hit(ray_index, ray, ray_data, leaf.first_primitive() + i, prim, lane,
                   sm_ptr_user);
        }
    }
}

template <typename Primitive, typename RayData, typename Intersection,
          typename OnHit>
GRACE_DEVICE
void leaf_intersector_rayloop_sm20(
    const CudaBvhLeaf& leaf, const Primitive* const leaf_prims, const Ray* rays,
    const int first_ray_index, RayData* rays_data, Intersection intersect,
    OnHit on_hit, const BoundedPtr<char>& sm_ptr_user)
{
    const int lane = threadIdx.x % grace::WARP_SIZE;

    for (int i = lane; i < leaf.size(); i += grace::WARP_SIZE)
    {
        const Primitive prim = leaf_prims[i];

        for (int j = 0; j < grace::WARP_SIZE; ++j)
        {
            const Ray ray_j = rays[j];
            RayData ray_data_j = rays_data[j];

            bool hit = intersect(ray_j, prim, ray_data_j, lane, sm_ptr_user);
            if (hit)
            {
                on_hit(first_ray_index + j, ray_j, ray_data_j, leaf.first_primitive() + i, prim,
                       lane, sm_ptr_user);
            }

            // There is an ambiguity for which ray data we update to.
            // We choose the highest-valued active lane's version.
            unsigned int hitbits = __ballot(hit);
            if (hitbits)
            {
                int high_lane = grace::WARP_SIZE - __clz(hitbits) - 1;
                GRACE_ASSERT(__popc(__ballot(lane == high_lane)) == 1);
                if (lane == high_lane)
                {
                    rays_data[j] = ray_data_j;
                }
            }
        }
    }
}

template <typename Primitive, typename RayData, typename Intersection,
          typename OnHit>
GRACE_DEVICE
void leaf_intersector_rayloop_sm30(
    const CudaBvhLeaf& leaf, const Primitive* const leaf_prims, const Ray& ray,
    const int first_ray_index, RayData& ray_data, Intersection intersect,
    OnHit on_hit, const BoundedPtr<char>& sm_ptr_user)
{
    const int lane = threadIdx.x % grace::WARP_SIZE;

    // All lanes have to hit the shfl_idx().
    const int n = (leaf.size() + grace::WARP_SIZE - 1) / grace::WARP_SIZE;
    for (int k = 0; k < n; ++k)
    {
        const int i = lane + k * grace::WARP_SIZE;
        Primitive prim;
        if (i < leaf.size()) prim = leaf_prims[i];

        for (int j = 0; j < grace::WARP_SIZE; ++j)
        {
            const Ray ray_j = shfl_idx(ray, j);
            RayData ray_data_j = shfl_idx(ray_data, j);

            bool hit = false;
            if (i < leaf.size())
            {
                hit = intersect(ray_j, prim, ray_data_j, lane, sm_ptr_user);
                if (hit)
                {
                    on_hit(first_ray_index + j, ray_j, ray_data_j, leaf.first_primitive() + i,
                           prim, lane, sm_ptr_user);
                }
            }

            // There is an ambiguity for which ray data we update to.
            // We choose the highest-valued active lane's version.
            unsigned int hitbits = __ballot(hit);
            if (hitbits)
            {
                int high_lane = grace::WARP_SIZE - __clz(hitbits) - 1;
                GRACE_ASSERT(__popc(__ballot(lane == high_lane)) == 1);
                // All lanes must take part.
                ray_data_j = shfl_idx(ray_data_j, high_lane);
                if (lane == j)
                {
                    ray_data = ray_data_j;
                }
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
          typename NodesIter,
          typename LeavesIter,
          typename PrimitiveIter,
          typename Init,
          typename Intersection, typename OnHit,
          typename OnRayEntry, typename OnRayExit>
__global__ void trace_kernel(
    RayIter rays,
    const size_t n_rays,
    const NodesIter nodes,
    const size_t n_nodes,
    const LeavesIter leaves,
    const int root,
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

        ray_entry(ray_index, ray, ray_data, sm_iter_usr);

    #if __CUDA_ARCH__ < 300
        if (LTConfig == LeafTraversal::ParallelPrimitives)
        {
            sm_rays[lane] = ray;
            sm_rays_data[lane] = ray_data;
        }
    #endif

        Vector3f invd(1.f / ray.dx, 1.f / ray.dy, 1.f / ray.dz);
        Vector3f origin(ray.ox, ray.oy, ray.oz);

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
                GRACE_ASSERT((*stack_ptr) < n_nodes);

                // Pop stack.
                CudaBvhNode node = nodes[*stack_ptr];
                stack_ptr--;

                GRACE_ASSERT(node.left_child() >= 0);
                GRACE_ASSERT(node.right_child() > 0);
                // Recall that leaf indices are offset by += n_nodes.
                GRACE_ASSERT(node.left_child() < 2 * n_nodes);
                GRACE_ASSERT(node.right_child() <= 2 * n_nodes);

                int lr_hit = AABBs_hit(invd, origin, ray.start, ray.end,
                                       node.left_AABB(), node.right_AABB());

                if (__any(lr_hit & 1u))
                {
                    stack_ptr++;
                    *stack_ptr = node.right_child();
                }
                if (__any(lr_hit >= 2))
                {
                    stack_ptr++;
                    *stack_ptr = node.left_child();
                }

                // FIXME: Produces compile-time warning.
                // See http://stackoverflow.com/questions/1712713/
                GRACE_ASSERT(stack_ptr < sm_stacks + grace::STACK_SIZE * (wid + 1), trace_stack_overflow);
            }

            while (*stack_ptr >= n_nodes && *stack_ptr >= 0)
            {
                // Pop stack.
                CudaBvhLeaf leaf = leaves[(*stack_ptr)-n_nodes];
                GRACE_ASSERT(((*stack_ptr) - n_nodes) < n_nodes + 1);
                stack_ptr--;

                GRACE_ASSERT(leaf.first_primitive() >= 0);
                GRACE_ASSERT(leaf.size() > 0);
                GRACE_ASSERT(leaf.first_primitive() + leaf.size() - 1 < n_primitives);

                for (int i = lane; i < leaf.size(); i += grace::WARP_SIZE)
                {
                    sm_prims[i] = primitives[leaf.first_primitive() + i];
                }

                if (LTConfig == LeafTraversal::ParallelRays)
                {
                    leaf_intersector(leaf, sm_prims, ray, ray_index, ray_data,
                                     intersect, on_hit, sm_iter_usr);
                }
                else if (LTConfig == LeafTraversal::ParallelPrimitives)
                {
                    int first_ray_index = ray_index - lane;
                #if __CUDA_ARCH__ < 300
                    leaf_intersector_rayloop_sm20(leaf,
                                                  sm_prims,
                                                  sm_rays,
                                                  first_ray_index,
                                                  sm_rays_data,
                                                  intersect,
                                                  on_hit,
                                                  sm_iter_usr);
                #else
                    leaf_intersector_rayloop_sm30(leaf,
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
    const CudaBvh& d_bvh,
    const size_t user_smem_bytes,
    Init init,
    Intersection intersect,
    OnHit on_hit,
    OnRayEntry ray_entry,
    OnRayExit ray_exit)
{
    typedef typename std::iterator_traits<PrimitiveIter>::value_type TPrimitive;
    TexRefIter<detail::CudaBvhNode, NODE_TEX_UID> nodes_iter;
    TexRefIter<detail::CudaBvhLeaf, LEAF_TEX_UID> leaves_iter;

    detail::Bvh_const_ref<CudaBvh> bvh_ref(d_bvh);

    size_t N_nodes = d_bvh.num_nodes();

    if (N_rays % grace::WARP_SIZE != 0) {
        std::stringstream msg_stream;
        msg_stream << "Number of rays must be a multiple of the warp size ("
                   << grace::WARP_SIZE << ").";
        const std::string msg = msg_stream.str();

        throw std::invalid_argument(msg);
    }

    detail::CudaBvhNode* nodes_ptr = thrust::raw_pointer_cast(bvh_ref.nodes().data());
    detail::CudaBvhLeaf* leaves_ptr = thrust::raw_pointer_cast(bvh_ref.leaves().data());

    cudaError_t cuerr;
    cuerr = nodes_iter.bind(nodes_ptr,
                            d_bvh.num_nodes() * sizeof(detail::CudaBvhNode));
    GRACE_CUDA_CHECK(cuerr);

    cuerr = leaves_iter.bind(leaves_ptr,
                             d_bvh.num_leaves() * sizeof(detail::CudaBvhLeaf));
    GRACE_CUDA_CHECK(cuerr);

    const int NT = grace::TRACE_THREADS_PER_BLOCK;
    const int blocks = std::min((int)((N_rays + NT - 1) / NT),
                                grace::MAX_BLOCKS);
    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    size_t sm_size = sizeof(TPrimitive) * d_bvh.max_per_leaf() * N_warps
                     + GRACE_ALIGNOF(TPrimitive) - 1 // For alignment correction.
                     + sizeof(int) * grace::STACK_SIZE * N_warps
                     + user_smem_bytes;

    if (LTConfig == LeafTraversal::ParallelPrimitives)
    {
        int device;
        cudaDeviceProp prop;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        if (prop.major < 3) {
            sm_size += sizeof(Ray) * grace::WARP_SIZE * N_warps;
            sm_size += GRACE_ALIGNOF(Ray) - 1;
            sm_size += sizeof(RayData) * grace::WARP_SIZE * N_warps;
            sm_size += GRACE_ALIGNOF(RayData) - 1;
        }
    }

    detail::trace_kernel<RayData, LTConfig><<<blocks, NT, sm_size>>>(
        d_rays_iter,
        N_rays,
        nodes_iter,
        N_nodes,
        leaves_iter,
        d_bvh.root_index(),
        d_prims_iter,
        N_primitives,
        d_bvh.max_per_leaf(),
        user_smem_bytes,
        init,
        intersect,
        on_hit,
        ray_entry,
        ray_exit);
    GRACE_CUDA_KERNEL_CHECK();

    GRACE_CUDA_CHECK(nodes_iter.unbind());
    GRACE_CUDA_CHECK(leaves_iter.unbind());
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
    const CudaBvh& d_bvh,
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
                             d_primitives.size(), d_bvh, user_smem_bytes, init,
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
    const CudaBvh& d_bvh,
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
                             d_bvh, user_smem_bytes, init, intersect, on_hit,
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
    const CudaBvh& d_bvh,
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
                                    d_primitives.size(), d_bvh,
                                    user_smem_bytes, init, intersect, on_hit,
                                    ray_entry, ray_exit);
}

} // namespace grace
