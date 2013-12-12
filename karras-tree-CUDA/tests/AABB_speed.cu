#include <sstream>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../types.h"
#include "../nodes.h"
#include "../ray.h"
#include "utils.cuh"
#include "../kernels/bintree_trace.cuh"

__global__ void AABB_multi_hit(const grace::Ray* rays,
                               const grace::Node* nodes,
                               const int N_rays,
                               const int N_AABBs)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    bool hit;

    while (tid < N_rays)
    {
        for (int i=0; i<N_AABBs; i++) {
            hit = grace::AABB_hit(rays[tid], nodes[i]);
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
    }

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

    // Perform ray-box intersection tests on CPU.
    double t = (double)clock() / CLOCKS_PER_SEC;
    for (int i=0; i<N_rays; i++) {
        for (int j=0; j<N_AABBs; j++) {
            bool hit = grace::AABB_hit(h_rays[i], h_nodes[j]);
        }
    }
    t = (double)clock() / CLOCKS_PER_SEC - t;

    thrust::device_vector<grace::Ray> d_rays = h_rays;
    thrust::device_vector<grace::Node> d_nodes = h_nodes;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AABB_multi_hit<<<48, 512>>>(thrust::raw_pointer_cast(d_rays.data()),
                                thrust::raw_pointer_cast(d_nodes.data()),
                                                         N_rays,
                                                         N_AABBs);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << N_rays << " rays tested against " << N_AABBs << " AABBs in "
              << std::endl;
    std::cout << "  i) CPU: " << t << " s." << std::endl;
    std::cout << " ii) GPU: " << elapsed << " ms." << std::endl;


}
