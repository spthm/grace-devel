#include "tris_trace.cuh"

#include "grace/cuda/functors/trace.cuh"
#include "grace/cuda/kernels/bintree_trace.cuh"

void trace_closest_tri(
    const thrust::device_vector<grace::Ray>& d_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    thrust::device_vector<int>& d_closest_tri_idx)
{
    grace::trace_texref<RayData_tri>(
        d_rays,
        d_tris,
        d_tree,
        0,
        grace::Init_null(),
        RayIntersect_tri(),
        OnHit_tri(),
        RayEntry_tri(),
        // This copies RayData_tri.data to the provided array for each ray.
        grace::RayExit_to_array<int>(
            thrust::raw_pointer_cast(d_closest_tri_idx.data()))
    );
}
