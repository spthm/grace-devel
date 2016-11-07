#include "tris_render.cuh"
#include "tris_trace.cuh"

#include "helper/vector_math.cuh"

#include "grace/cuda/kernel_config.h"

#include <algorithm>
#include <cmath>

void setup_lights(
    const float3 bots, const float3 tops,
    thrust::device_vector<float3>& d_lights_pos)
{
    float3 centre = make_float3((bots.x + tops.x) / 2.,
                                (bots.y + tops.y) / 2.,
                                (bots.z + tops.z) / 2.);
    float max_span = max(tops.x - bots.x,
                         max(tops.y - bots.y, tops.z - bots.z));

    // Above
    d_lights_pos.push_back(
        make_float3(centre.x, tops.y + max_span, tops.z + max_span)
    );
    // Left
    // d_lights_pos.push_back(
    //     make_float3(bots.x - max_span, centre.y, tops.z + max_span)
    // );
}

void setup_camera(
    const float3 bots, const float3 tops, const float FOVy_degrees,
    const int resolution_x, const int resolution_y,
    float3* camera_position, float3* look_at, float3* view_up,
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
    float L_x = 1.02 * size.x / FOVx_radians;
    float L_y = 1.02 * size.y / *FOVy_radians;
    float camera_z = look_at->z + std::max(L_x, L_y);

    *camera_position = make_float3(bots.x - 0.1 * size.x,
                                   tops.y + 0.3 * size.y,
                                   camera_z);
}

static __global__ void shade_triangles_kernel(
    const Triangle* triangles,
    const size_t N_tris,
    const float3* const lights_pos,
    const size_t N_lights,
    float* const shaded_tris)
{
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < N_tris;
         tid += gridDim.x)
    {
        Triangle tri = triangles[tid];
        float3 normal = normalize(cross_product(tri.e1, tri.e2));

        for (int l = 0; l < N_lights; ++l)
        {
            float3 light_pos = lights_pos[l];
            float3 L = normalize(light_pos - tri.v);

            // The true value would vary with the point at which the ray
            // intersects the triangle. However, provided that
            // |L| >> |tri.e1|, |tri.e2| (i.e. the light is far away from the
            // triangle) the below is approximately correct.
            float shading = max(0.0, dot_product(L, normal));

            shaded_tris[l * N_tris + tid] = shading;
        }
    }
}

void shade_triangles(
    const thrust::device_vector<Triangle>& d_tris,
    const thrust::device_vector<float3>& d_lights_pos,
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
    const grace::Tree& d_tree,
    const thrust::device_vector<float3>& d_lights_pos,
    const thrust::device_vector<float>& d_shaded_tris,
    thrust::device_vector<float>& d_pixels)
{
    thrust::device_vector<grace::Ray> d_shadow_rays(d_rays.size());
    thrust::device_vector<PrimaryRayResult> d_primary_results(d_rays.size());
    thrust::device_vector<ShadowRayResult>
        d_shadow_results(d_rays.size() * d_lights_pos.size());

    trace_primary_rays(d_rays, d_tris, d_tree, d_primary_results);

    // Trace shadow rays to each light source.
    for (int i = 0; i < d_lights_pos.size(); ++i)
    {
        generate_shadow_rays(i, d_lights_pos, d_rays, d_primary_results,
                             d_shadow_rays);

        trace_shadow_rays(d_shadow_rays, d_tris, d_tree,
                          d_shadow_results.data() + i * d_rays.size());
    }

    shade_pixels(d_primary_results, d_shadow_results, d_shaded_tris, d_pixels);
}
