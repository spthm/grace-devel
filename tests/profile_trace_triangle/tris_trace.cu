#include "tris_trace.cuh"

#include "grace/cuda/detail/functors/trace.cuh"
#include "grace/cuda/detail/kernels/bintree_trace.cuh"


void setup_cameras(
    const grace::AABB<float> aabb,
    const float FOVy_degrees,
    const int resolution_x, const int resolution_y,
    std::vector<grace::Vector<3, float> >& camera_positions,
    grace::Vector<3, float>* look_at,
    grace::Vector<3, float>* view_up,
    float* FOVy_radians, float* ray_length)
{
    const grace::Vector<3, float> size = aabb.size();
    const grace::Vector<3, float> center = aabb.center();

    *look_at = center;
    *view_up = grace::Vector<3, float>(0.f, 1.f, 0.f);
    *ray_length = 100. * size.z;

    *FOVy_radians = FOVy_degrees * 3.141 / 180.;

    // Compute the z-position of the camera, given the fixed field-of-view, such
    // that the entire bounding box will always be visible.
    float FOVx_radians = 2. * std::atan2(std::tan(*FOVy_radians / 2.),
                                         (double)resolution_x / resolution_y);
    float L_x = 1.1 * size.x / FOVx_radians;
    float L_y = 1.1 * size.y / *FOVy_radians;
    float camera_z = look_at->z + std::max(L_x, L_y);

    camera_positions.push_back(grace::Vector<3, float>(aabb.min.x - 0.1 * size.x,
                                                       aabb.max.y + 0.3 * size.y,
                                                       camera_z));
    camera_positions.push_back(grace::Vector<3, float>(aabb.max.x + 0.1 * size.x,
                                                       aabb.min.y - 0.3 * size.y,
                                                       camera_z));
    camera_positions.push_back(grace::Vector<3, float>(center.x, center.y, camera_z));
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
