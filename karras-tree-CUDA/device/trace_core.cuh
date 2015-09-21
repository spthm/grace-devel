#pragma once

#include "../types.h"
#include "trace_functors.cuh"
#include "../util/bound_iter.cuh"

namespace grace {

namespace gpu {

// We must forward declare trace_kernel so we can make it a friend of TraceCore.
// The third template parameter is not constrained to be an instantiaion of
// the below TraceCore because trace_kernel does not actually enforce this.
// Trying to enforce it within TraceCore would require a partial specialization
// of trace_kernel, which is not allowed.
template <
    typename RayIter,
    typename PrimitiveIter,
    typename Core
>
__global__ void trace_kernel(const RayIter, const size_t, const float4*,
                             const size_t, const int4*, const int*,
                             const PrimitiveIter, const size_t, const int,
                             const size_t, Core);

// The redundant method wrapping present here is done so that
//   1. Policy classes may all implement operator(), rather than e.g.
//      init_impl(), ray_entry_impl() ...
//   2. Method call syntax is cleaner; it would otherwise be
//      core.BasePolicy::operator()(args...)
//   3. We may explicitly define the requirements of the TraceCore interface.
// A consequence of the third point is that any compile-time errors in a
// client's implementation of the interface will ultimately point back to this
// class, rather than to where they are called in the trace kernel, which should
// be more helpful.
// RayDataPOD MUST, as the name implies, be a POD class or struct.
// RayDataPOD will be zero-initialized for each ray.
template <
    typename RayDataPOD,
    typename IntersectionPolicy,
    typename OnHitPolicy,
    typename RayEntryPolicy = RayEntry_null,
    typename RayExitPolicy = RayExit_null,
    typename InitPolicy = Init_null
>
class TraceCore : public IntersectionPolicy,
                  public OnHitPolicy,
                  public RayEntryPolicy,
                  public RayExitPolicy,
                  public InitPolicy
{
    // C++11
    // static_assert(std::is_pod<RayDataPOD>::value, "RayDataPOD must be a POD type.");
public:
    typedef RayDataPOD RayData;

private:
    // trace_kernel must already be declared.
    template <typename RayIter, typename PrimitiveIter, typename Core>
    friend void trace_kernel(const RayIter, const size_t, const float4*,
                             const size_t, const int4*, const int*,
                             const PrimitiveIter, const size_t, const int,
                             const size_t, Core);

    GRACE_DEVICE void init(BoundIter<char> smem_iter)
    {
        InitPolicy::operator()(smem_iter);
    }

    GRACE_DEVICE void ray_entry(const int ray_idx,
                                const Ray& ray,
                                RayDataPOD* const ray_data_ptr,
                                BoundIter<char> smem_iter)
    {
        RayEntryPolicy::operator()(ray_idx, ray, ray_data_ptr, smem_iter);
    }

    GRACE_DEVICE void ray_exit(const int ray_idx,
                               const Ray& ray,
                               RayDataPOD* const ray_data_ptr,
                               BoundIter<char> smem_iter)
    {
        RayExitPolicy::operator()(ray_idx, ray, ray_data_ptr, smem_iter);
    }

    template <typename TPrimitive>
    GRACE_DEVICE bool intersect(const int ray_idx,
                                const Ray& ray,
                                RayDataPOD* const ray_data_ptr,
                                const int primitive_idx,
                                const TPrimitive& primitive,
                                const int lane,
                                BoundIter<char> smem_iter)
    {
        bool hit = IntersectionPolicy::operator()(ray_idx, ray, ray_data_ptr,
                                                  primitive_idx, primitive,
                                                  lane, smem_iter);
        return hit;
    }

    template <typename TPrimitive>
    GRACE_DEVICE void on_hit(const int ray_idx,
                             const Ray& ray,
                             RayDataPOD* const ray_data_ptr,
                             const int primitive_idx,
                             const TPrimitive& primitive,
                             const int lane,
                             BoundIter<char> smem_iter)
    {
        OnHitPolicy::operator()(ray_idx, ray, ray_data_ptr,
                                primitive_idx, primitive,
                                lane, smem_iter);
    }
};

} // namespace gpu

} // namespace grace
