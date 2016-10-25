// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "triangle.cuh"
#include "tris_tree.cuh"
#include "tris_trace.cuh"

#include "grace/cuda/nodes.h"
#include "grace/ray.h"
#include "helper/images.hpp"
#include "helper/rays.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;

    size_t N_rays = 1024 * 1024; // = 1,048,576
    int max_per_leaf = 32;
    std::string fname = "../data/ply/dragon_recon/dragon_vrip.ply";
    unsigned int device_ID = 0;

    if (argc > 1) {
        N_rays = 32 * (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        max_per_leaf = (int)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        fname = std::string(argv[3]);
    }
    if (argc > 4) {
        device_ID = (unsigned int)std::strtol(argv[4], NULL, 10);
    }

    cudaGetDeviceProperties(&deviceProp, device_ID);
    cudaSetDevice(device_ID);

    size_t N_per_side = std::floor(std::pow(N_rays, 0.500001));
    // N_rays must be a multiple of 32.
    N_per_side = ((N_per_side + 32 - 1) / 32) * 32;
    N_rays = N_per_side * N_per_side;

    std::cout << "Input geometry file:     " << fname << std::endl;
    // Vector is resized in read_triangles().
    std::vector<PLYTriangle> ply_tris;
    thrust::device_vector<Triangle> d_tris;
    read_triangles(fname, ply_tris);
    d_tris = ply_tris;
    const size_t N = d_tris.size();

    std::cout << "Number of primitives:    " << N << std::endl
              << "Number of rays:          " << N_rays << std::endl
              << "Max primitives per leaf: " << max_per_leaf << std::endl
              << "Running on device:       " << device_ID
                                             << " (" << deviceProp.name << ")"
                                             << std::endl
              << std::endl;

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    float3 bots, tops;
    grace::Tree d_tree(N, max_per_leaf);
    build_tree_tris(d_tris, d_tree, &bots, &tops);

    // maxs.w is padding to move ray-generating plane away from model AABB.
    // Applies equally to x/y/z bounds.
    float padding = 0.05 * max(tops.x - bots.y,
                               max(tops.y - bots.y, tops.z - bots.z));
    float4 mins = make_float4(bots.x, bots.y, bots.z, padding);
    float4 maxs = make_float4(tops.x, tops.y, tops.z, padding);

    orthogonal_rays_z(N_per_side, mins, maxs, d_rays);

    thrust::device_vector<float3> d_lights_pos;
    setup_lights(bots, tops, d_lights_pos);

    thrust::device_vector<float> d_brightness(N_rays);
    trace_shade_tri(d_rays,
                    d_tris,
                    d_tree,
                    d_lights_pos,
                    d_brightness);

    float min_brightness = thrust::reduce(d_brightness.begin(),
                                          d_brightness.end(),
                                          1E20f, thrust::minimum<float>());
    float max_brightness = thrust::reduce(d_brightness.begin(),
                                          d_brightness.end(),
                                          0.f, thrust::maximum<float>());
    std::cout << "Minimum brightness: " << min_brightness << std::endl
              << "Maximum brightness: " << max_brightness << std::endl;

    thrust::host_vector<float> h_brightness = d_brightness;
    make_bitmap(thrust::raw_pointer_cast(h_brightness.data()),
                N_per_side, N_per_side,
                0.f, d_lights_pos.size() * 1.f + AMBIENT_BKG,
                "render.bmp",
                255, 255, 255);

    return EXIT_SUCCESS;
}
