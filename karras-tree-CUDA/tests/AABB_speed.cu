#include <sstream>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../types.h"
#include "../nodes.h"
#include "../ray.h"
#include "utils.cuh"
#include "../kernels/bintree_trace.cuh"
#include "../kernels/morton.cuh"

__global__ void AABB_hit_eisemann_kernel(const grace::Ray* rays,
                                         const grace::Node* nodes,
                                         const int N_rays,
                                         const int N_AABBs,
                                         unsigned int* hits)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        grace::Ray ray = rays[tid];

        float xbyy = ray.dx / ray.dy;
        float ybyx = 1.0f / xbyy;
        float ybyz = ray.dy / ray.dz;
        float zbyy = 1.0f / ybyz;
        float xbyz = ray.dx / ray.dz;
        float zbyx = 1.0f / xbyz;

        float c_xy = ray.oy - ybyx*ray.ox;
        float c_xz = ray.oz - zbyx*ray.ox;
        float c_yx = ray.ox - xbyy*ray.oy;
        float c_yz = ray.oz - zbyy*ray.oy;
        float c_zx = ray.ox - xbyz*ray.oz;
        float c_zy = ray.oy - ybyz*ray.oz;

        for (int i=0; i<N_AABBs; i++) {
            if (grace::AABB_hit_eisemann(ray, nodes[i],
                                         xbyy, ybyx, ybyz, zbyy, xbyz, zbyx,
                                         c_xy, c_xz, c_yx, c_yz, c_zx, c_zy))
                hits[tid]++;
        }

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void AABB_hit_plucker_kernel(const grace::Ray* rays,
                                        const grace::Node* nodes,
                                        const int N_rays,
                                        const int N_AABBs,
                                        unsigned int* hits)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        for (int i=0; i<N_AABBs; i++) {
            if (grace::AABB_hit_plucker(rays[tid], nodes[i]))
                hits[tid]++;
        }

        tid += blockDim.x * gridDim.x;
    }
}

int main(void)
{
    // Init.
    int N_rays = 100000;
    int N_AABBs = 500;

    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<grace::Node> h_nodes(N_AABBs);
    thrust::host_vector<UInteger32> h_keys(N_rays);

    // Generate rays
    random_float_functor rng(-1.0f, 1.0f);
    float x, y, z, N;
    for (int i=0; i<N_rays; i++) {
        // RNG in (-1, 1).
        x = rng(3*i+0);
        y = rng(3*i+1);
        z = rng(3*i+2);

        N = sqrt(x*x + y*y + z*z);

        h_rays[i].dx = x / N;
        h_rays[i].dy = y / N;
        h_rays[i].dz = z / N;

        h_rays[i].ox = h_rays[i].oy = h_rays[i].oz = 0;

        h_rays[i].length = N;

        h_rays[i].dclass = 0;
        if (x >= 0)
            h_rays[i].dclass += 1;
        if (y >= 0)
            h_rays[i].dclass += 2;
        if (z >= 0)
            h_rays[i].dclass += 4;

        h_keys[i] = grace::morton_key((h_rays[i].dx+1)/2.f,
                                      (h_rays[i].dy+1)/2.f,
                                      (h_rays[i].dz+1)/2.f);
    }
    // Sort rays by Morton key.
    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    h_keys.clear();
    h_keys.shrink_to_fit();

    // Generate AABBs
    float x1, x2, y1, y2, z1, z2;
    for (int i=0; i<N_AABBs; i++) {
        x1 = rng(3*N_rays + 6*i+0);
        x2 = rng(3*N_rays + 6*i+1);
        y1 = rng(3*N_rays + 6*i+2);
        y2 = rng(3*N_rays + 6*i+3);
        z1 = rng(3*N_rays + 6*i+4);
        z2 = rng(3*N_rays + 6*i+5);

        h_nodes[i].top[0] = max(x1, x2);
        h_nodes[i].top[1] = max(y1, y2);
        h_nodes[i].top[2] = max(z1, z2);

        h_nodes[i].bottom[0] = min(x1, x2);
        h_nodes[i].bottom[1] = min(y1, y2);
        h_nodes[i].bottom[2] = min(z1, z2);
    }

    thrust::device_vector<grace::Ray> d_rays = h_rays;
    thrust::device_vector<grace::Node> d_nodes = h_nodes;
    thrust::device_vector<unsigned int> d_hits(N_rays);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AABB_hit_eisemann_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_nodes.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_hits.data()));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << N_rays << " rays tested against " << N_AABBs
              << " AABBs (ray slopes) in" << std::endl;
    std::cout << "   GPU: " << elapsed << " ms." << std::endl;
    std::cout << d_hits[0] << ", " << d_hits[50000-1]
              << ", " << d_hits[100000-1] << std::endl;


    // Perform plucker ray-box intersection tests on CPU.
    // thrust::fill(h_hits.begin(), h_hits.end(), 0u);
    // t = (double)clock() / CLOCKS_PER_SEC;
    // for (int i=0; i<N_rays; i++) {
    //     for (int j=0; j<N_AABBs; j++) {
    //         if (grace::AABB_hit_plucker(h_rays[i], h_nodes[j]))
    //             h_hits[i]++;
    //     }
    // }
    // t = (double)clock() / CLOCKS_PER_SEC - t;

    thrust::fill(d_hits.begin(), d_hits.end(), 0u);
    cudaEventRecord(start);
    AABB_hit_plucker_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_nodes.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_hits.data()));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << N_rays << " rays tested against " << N_AABBs
              << " AABBs (plucker) in" << std::endl;
    // std::cout << "  i) CPU: " << t*1000. << " ms." << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;
    std::cout << d_hits[0] << ", " << d_hits[50000-1]
              << ", " << d_hits[100000-1] << std::endl;

}
