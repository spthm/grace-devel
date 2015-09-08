#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#include <thrust/device_vector.h>

#include "../device/intersect.cuh"
#include "../error.h"
#include "../device/util.cuh"
#include "../kernel_config.h"
#include "../nodes.h"
#include "../ray.h"
#include "../types.h"

#ifdef GRACE_NODES_TEX
#define FETCH_NODE(array, i) tex1Dfetch(array##_tex, i)
#else
#define FETCH_NODE(array, i) array[i]
#endif

#ifdef GRACE_PRIMITIVES_TEX
#define FETCH_PRIMITIVE(array, i) tex1Dfetch(array##_tex, i)
#else
#define FETCH_PRIMITIVE(array, i) array[i]
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

// FIXME: this now depends on the templated type of the primitives.
//        can it be placed within a function?
#ifdef GRACE_PRIMITIVES_TEX
texture<float4, cudaTextureType1D, cudaReadModeElementType> primitives_tex;
#endif


namespace gpu {

//-----------------------------------------------------------------------------
// CUDA tracing kernel.
//-----------------------------------------------------------------------------

template <typename RayData,
          typename TPrimitive,
          typename Init,
          typename Intersection, typename OnHit,
          typename OnRayEntry, typename OnRayExit>
__global__ void trace_kernel(
    const Ray* rays,
    const size_t n_rays,
    const float4* nodes,
    const size_t n_nodes,
    const int4* leaves,
    const int* root_index,
    const TPrimitive* primitives, // Pointer to primitive spatial-data wrapper type
    const size_t n_primitives,
    const int max_per_leaf,
    const size_t user_smem_bytes, // User's SMEM allocation, in bytes.
    Init init,              // pre-traversal functor
    Intersection intersect, // ray-primitive intersection test functor
    OnHit on_hit,           // ray-primitive intersection processing functor
    OnRayEntry ray_entry,   // ray-traversal entry functor
    OnRayExit ray_exit)     // ray-traversal exit functor
{
    const int lane = threadIdx.x % grace::WARP_SIZE;
    const int wid  = threadIdx.x / grace::WARP_SIZE;
    int ray_index = threadIdx.x + blockIdx.x * blockDim.x;

    // The tree's root index can be anywhere in ALBVH.
    const int root = *root_index;

    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t prims_smem_count = max_per_leaf * N_warps;

    extern __shared__ char smem_trace[];
    const UserSmemPtr<char> sm_ptr_usr(smem_trace, user_smem_bytes);
    init(sm_ptr_usr);
    __syncthreads();

    // Shared memory accesses must ensure correct alignment relative to the
    // access size.
    char* sm_ptr = smem_trace + user_smem_bytes;
    int rem = (uintptr_t)sm_ptr % sizeof(TPrimitive);
    if (rem != 0) {
        sm_ptr += sizeof(TPrimitive) - rem;
    }
    TPrimitive* sm_prims = reinterpret_cast<TPrimitive*>(sm_ptr);
    int* sm_stacks = reinterpret_cast<int*>(&sm_prims[prims_smem_count]);
    int* stack_ptr = sm_stacks + grace::STACK_SIZE * wid;

    // This is the exit sentinel. All threads in a ray packet (i.e. warp) write
    // to the same location to avoid any need for volatile declarations, or
    // warp-synchronous instructions (as far as the stack is concerned).
    *stack_ptr = -1;

    while (ray_index < n_rays)
    {
        // Ray must not be modified by user.
        const Ray ray = rays[ray_index];
        RayData ray_data;
        ray_entry(ray_index, ray, ray_data, sm_ptr_usr);

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

                int lr_hit = AABBs_hit(invd, origin, ray.length,
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
                GRACE_ASSERT(stack_ptr < sm_stacks + grace::STACK_SIZE * (wid + 1) && "trace stack overflowed");
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
                    sm_prims[max_per_leaf * wid + i]
                        = FETCH_PRIMITIVE(primitives, node.x + i);
                }

                for (int i = 0; i < node.y; ++i)
                {
                    TPrimitive prim = sm_prims[max_per_leaf * wid + i];
                    if (intersect(ray, prim, ray_data, i, sm_ptr_usr))
                    {
                        on_hit(ray_index, ray, ray_data, node.x + i, prim, i,
                               sm_ptr_usr);
                    }
                }
            }
        }
        ray_exit(ray_index, ray, ray_data, sm_ptr_usr);
        ray_index += blockDim.x * gridDim.x;
    }
}

} // namespace gpu

//-----------------------------------------------------------------------------
// C-like wrapper for trace kernel.
//-----------------------------------------------------------------------------

template <typename RayData,
          typename TPrimitive,
          typename Init,
          typename Intersection, typename OnHit,
          typename OnRayEntry, typename OnRayExit>
GRACE_HOST void trace(
    const thrust::device_vector<Ray>& d_rays,
    const thrust::device_vector<TPrimitive> d_primitives,
    const Tree& d_tree,
    const size_t user_smem_bytes,
    Init init,
    Intersection intersect,
    OnHit on_hit,
    OnRayEntry ray_entry,
    OnRayExit ray_exit)
{
    size_t n_rays = d_rays.size();
    size_t n_nodes = d_tree.leaves.size() - 1;

    if (n_rays % grace::WARP_SIZE != 0) {
        std::stringstream msg_stream;
        msg_stream << "Number of rays must be a multiple of the warp size ("
                   << grace::WARP_SIZE << ").";
        const std::string msg = msg_stream.str();

        throw std::invalid_argument(msg);
    }

#if defined(GRACE_NODES_TEX) || defined(GRACE_PRIMITIVES_TEX)
    cudaError_t cuerr;
#endif

#ifdef GRACE_NODES_TEX
    cuerr = cudaBindTexture(
        0, nodes_tex,
        reinterpret_cast<const float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        d_tree.nodes.size() * sizeof(float4));
    GRACE_CUDA_CHECK(cuerr);

    cuerr = cudaBindTexture(
        0, leaves_tex, thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.leaves.size() * sizeof(int4));
    GRACE_CUDA_CHECK(cuerr);
#endif

#ifdef GRACE_PRIMITIVES_TEX
    cuerr = cudaBindTexture(
        0, primitives_tex,
        thrust::raw_pointer_cast(d_primitives.data()),
        d_primitives.size() * sizeof(TPrimitive));
    GRACE_CUDA_CHECK(cuerr);
#endif

    const int NT = grace::TRACE_THREADS_PER_BLOCK;
    const int blocks = min(static_cast<int>((n_rays + NT - 1) / NT),
                           grace::MAX_BLOCKS);
    const size_t N_warps = grace::TRACE_THREADS_PER_BLOCK / grace::WARP_SIZE;
    const size_t sm_size = sizeof(TPrimitive) * d_tree.max_per_leaf * N_warps
                           + sizeof(TPrimitive) - 1 // For alignment correction.
                           + sizeof(int) * grace::STACK_SIZE * N_warps
                           + user_smem_bytes;

    gpu::trace_kernel<RayData><<<blocks, NT, sm_size>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        n_rays,
        reinterpret_cast<const float4*>(
            thrust::raw_pointer_cast(d_tree.nodes.data())),
        n_nodes,
        thrust::raw_pointer_cast(d_tree.leaves.data()),
        d_tree.root_index_ptr,
        thrust::raw_pointer_cast(d_primitives.data()),
        d_primitives.size(),
        d_tree.max_per_leaf,
        user_smem_bytes,
        init,
        intersect,
        on_hit,
        ray_entry,
        ray_exit);
    GRACE_KERNEL_CHECK();

#ifdef GRACE_NODES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(nodes_tex));
    GRACE_CUDA_CHECK(cudaUnbindTexture(leaves_tex));
#endif
#ifdef GRACE_PRIMITIVES_TEX
    GRACE_CUDA_CHECK(cudaUnbindTexture(primitives_tex));
#endif
}

} // namespace grace
