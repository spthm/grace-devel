#pragma once

#include "triangle.cuh"

#include "grace/aabb.h"
#include "grace/ray.h"
#include "grace/vector.h"
#include "grace/cuda/nodes.h"

#include <thrust/device_vector.h>

#define AMBIENT_BKG 0.1f
#define BKG_COLOUR 0.0f

void setup_lights(
    const grace::AABB<float> aabb,
    thrust::device_vector<grace::Vector<3, float> >& d_lights_pos);

void setup_camera(
    const grace::AABB<float> aabb,
    const float FOVy_degrees,
    const int resolution_x, const int resolution_y,
    grace::Vector<3, float>* camera_position,
    grace::Vector<3, float>* look_at,
    grace::Vector<3, float>* view_up,
    float* FOVy_radians, float* ray_length);

void shade_triangles(
    const thrust::device_vector<Triangle>& d_tris,
    const thrust::device_vector<grace::Vector<3, float> >& d_lights_pos,
    thrust::device_vector<float>& d_shaded_tris);

void render(
    const thrust::device_vector<grace::Ray>& d_rays,
    const thrust::device_vector<Triangle>& d_tris,
    const grace::Tree& d_tree,
    const thrust::device_vector<grace::Vector<3, float> >& d_lights_pos,
    const thrust::device_vector<float>& d_shaded_tris,
    thrust::device_vector<float>& d_pixels);
