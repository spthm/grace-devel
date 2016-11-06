#include "tris_trace.cuh"

#include "grace/cuda/functors/trace.cuh"
#include "grace/cuda/kernels/bintree_trace.cuh"


void setup_cameras(
    const float3 bots, const float3 tops, const float FOVy_degrees,
    const int resolution_x, const int resolution_y,
    std::vector<float3>& camera_positions, float3* look_at, float3* view_up,
    float* FOVy_radians, float* ray_length)
{
    const float3 size = make_float3(tops.x - bots.x,
                                    tops.y - bots.y,
                                    tops.z - bots.z);
    const float3 center = make_float3((bots.x + tops.x) / 2.,
                                      (bots.y + tops.y) / 2.,
                                      (bots.z + tops.z) / 2.);

    *look_at = center;
    *view_up = make_float3(0.f, 1.f, 0.f);
    *ray_length = 100. * size.z;

    *FOVy_radians = FOVy_degrees * 3.141 / 180.;

    // Compute the z-position of the camera, given the fixed field-of-view, such
    // that the entire bounding box will always be visible.
    float FOVx_radians = 2. * std::atan2(std::tan(*FOVy_radians / 2.),
                                         (double)resolution_x / resolution_y);
    float L_x = 1.1 * size.x / FOVx_radians;
    float L_y = 1.1 * size.y / *FOVy_radians;
    float camera_z = look_at->z + std::max(L_x, L_y);

    camera_positions.push_back(make_float3(bots.x - 0.1 * size.x,
                                           tops.y + 0.3 * size.y,
                                           camera_z));
    camera_positions.push_back(make_float3(tops.x + 0.1 * size.x,
                                           bots.y - 0.3 * size.y,
                                           camera_z));
    camera_positions.push_back(make_float3(center.x, center.y, camera_z));
}

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
