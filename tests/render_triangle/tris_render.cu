#include "tris_render.cuh"
#include "tris_trace.cuh"

#include "grace/cuda/detail/kernel_config.h"

#include <algorithm>
#include <cmath>

void setup_lights(
    const grace::AABB<float> aabb,
    thrust::device_vector<grace::Vector<3, float> >& d_lights_pos)
{
    const grace::Vector<3, float> center = aabb.center();
    float max_span = max_element(aabb.size());

    // Above
    d_lights_pos.push_back(
        grace::Vector<3, float>(center.x,
                                aabb.max.y + max_span,
                                aabb.max.z + max_span)
    );
    // Left
    // d_lights_pos.push_back(
    //     grace::Vector<3, float>(aabb.min.x - max_span,
    //                             center.y,
    //                             aabb.max.z + max_span)
    // );
}

void setup_camera(
    const grace::AABB<float> aabb,
    const float FOVy_degrees,
    const int resolution_x, const int resolution_y,
    grace::Vector<3, float>* camera_position,
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
    float L_x = 1.02 * size.x / FOVx_radians;
    float L_y = 1.02 * size.y / *FOVy_radians;
    float camera_z = look_at->z + std::max(L_x, L_y);

    *camera_position = grace::Vector<3, float>(aabb.min.x - 0.1 * size.x,
                                               aabb.max.y + 0.3 * size.y,
                                               camera_z);
}

static __global__ void shade_triangles_kernel(
    const Triangle* triangles,
    const size_t N_tris,
    const grace::Vector<3, float>* const lights_pos,
    const size_t N_lights,
    float* const shaded_tris)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < N_tris;
         tid += gridDim.x)
    {
        Triangle tri = triangles[tid];
        grace::Vector<3, float> normal = normalize(cross(tri.e1, tri.e2));

        for (int l = 0; l < N_lights; ++l)
        {
            grace::Vector<3, float> light_pos = lights_pos[l];
            grace::Vector<3, float> L = normalize(light_pos - tri.v);

            // The true value would vary with the point at which the ray
            // intersects the triangle. However, provided that
            // |L| >> |tri.e1|, |tri.e2| (i.e. the light is far away from the
            // triangle) the below is approximately correct.
            float shading = max(0.0, dot(L, normal));

            shaded_tris[l * N_tris + tid] = shading;
        }
    }
}

void shade_triangles(
    const thrust::device_vector<Triangle>& d_tris,
    const thrust::device_vector<grace::Vector<3, float> >& d_lights_pos,
    thrust::device_vector<float>& d_shaded_tris)
{
    d_shaded_tris.resize(d_tris.size() * d_lights_pos.size());

    const int NT = 128;
    const int blocks = min((int)((d_tris.size() + NT - 1) / NT), grace::MAX_BLOCKS);
    shade_triangles_kernel<<<NT, blocks>>>(
        thrust::raw_pointer_cast(d_tris.data()),
        d_tris.size(),
        thrust::raw_pointer_cast(d_lights_pos.data()),
        d_lights_pos.size(),
        thrust::raw_pointer_cast(d_shaded_tris.data())
    );

}

static __global__ void shade_pixels_kernel(
    const PrimaryRayResult* const primary_results,
    const size_t N_primary,
    const ShadowRayResult* const shadow_results,
    const int N_shadow_per_primary,
    const float* const shaded_tris,
    float* const pixels)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < N_primary;
         tid += gridDim.x)
    {
        float brightness = 0.f;
        for (int i = 0; i < N_shadow_per_primary; ++i)
        {
            int tri_idx = shadow_results[i * N_shadow_per_primary + tid].idx;
            if (tri_idx == -1)
            {
                // Not blocked.
                brightness = brightness + 1.f;
            }
        }
        brightness = AMBIENT_BKG + brightness / N_shadow_per_primary;

        float colour = BKG_COLOUR; // Assume miss.
        int tri_idx = primary_results[tid].idx;
        if (tri_idx != -1)
        {
            colour = brightness * shaded_tris[tri_idx];
        }

        pixels[tid] = colour;
    }
}

static void shade_pixels(
    const thrust::device_vector<PrimaryRayResult>& d_primary_results,
    const thrust::device_vector<ShadowRayResult>& d_shadow_results,
    const thrust::device_vector<float>& d_shaded_tris,
    thrust::device_vector<float>& d_pixels)
{
    const int NT = 128;
    const int blocks = min((int)((d_primary_results.size() + NT - 1) / NT), grace::MAX_BLOCKS);
    shade_pixels_kernel<<<blocks, NT>>>(
        thrust::raw_pointer_cast(d_primary_results.data()),
        d_primary_results.size(),
        thrust::raw_pointer_cast(d_shadow_results.data()),
        d_shadow_results.size() / d_primary_results.size(),
        thrust::raw_pointer_cast(d_shaded_tris.data()),
        thrust::raw_pointer_cast(d_pixels.data())
    );
}

void render(
    const thrust::device_vector<grace::Ray>& d_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::CudaBvh& d_bvh,
    const thrust::device_vector<grace::Vector<3, float> >& d_lights_pos,
    const thrust::device_vector<float>& d_shaded_tris,
    thrust::device_vector<float>& d_pixels)
{
    thrust::device_vector<grace::Ray> d_shadow_rays(d_rays.size());
    thrust::device_vector<PrimaryRayResult> d_primary_results(d_rays.size());
    thrust::device_vector<ShadowRayResult>
        d_shadow_results(d_rays.size() * d_lights_pos.size());

    trace_primary_rays(d_rays, d_tris, d_bvh, d_primary_results);

    // Trace shadow rays to each light source.
    for (int i = 0; i < d_lights_pos.size(); ++i)
    {
        generate_shadow_rays(i, d_lights_pos, d_rays, d_primary_results,
                             d_shadow_rays);

        trace_shadow_rays(d_shadow_rays, d_tris, d_bvh,
                          d_shadow_results.data() + i * d_rays.size());
    }

    shade_pixels(d_primary_results, d_shadow_results, d_shaded_tris, d_pixels);
}
