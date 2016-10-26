#pragma once

#include "triangle.cuh"

#include "grace/ray.h"
#include "grace/cuda/nodes.h"

#include <thrust/device_vector.h>

#define AMBIENT_BKG 0.05f
#define BKG_COLOUR 0.0f

void setup_lights(
    const float3 bots, const float3 tops,
    thrust::device_vector<float3>& d_lights_pos);

void shade_triangles(
    const thrust::device_vector<Triangle>& d_tris,
    const thrust::device_vector<float3>& d_lights_pos,
    thrust::device_vector<float>& d_shaded_tris);

void render(
    const thrust::device_vector<grace::Ray>& d_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    const thrust::device_vector<float3>& d_lights_pos,
    const thrust::device_vector<float>& d_shaded_tris,
    thrust::device_vector<float>& d_pixels);
