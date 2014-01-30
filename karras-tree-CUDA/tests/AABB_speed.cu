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
                                         unsigned int* ray_hits,
                                         unsigned int* box_hits)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        grace::Ray ray = rays[tid];

        grace::SlopeProp slope = slope_properties(ray);

        for (int i=0; i<N_AABBs; i++) {
            if (grace::AABB_hit_eisemann(ray, slope, nodes[i]))
            {
                ray_hits[tid]++;
                //atomicAdd(&(box_hits[i]), 1);
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}

__host__ __device__ bool AABB_hit_plucker(const grace::Ray& ray,
                                          const grace::Node& node)
{
    float rx = ray.dx;
    float ry = ray.dy;
    float rz = ray.dz;
    float length = ray.length;
    float s2bx, s2by, s2bz; // Vector from ray start to lower cell corner.
    float s2tx, s2ty, s2tz; // Vector from ray start to upper cell corner.

    s2bx = node.bottom[0] - ray.ox;
    s2by = node.bottom[1] - ray.oy;
    s2bz = node.bottom[2] - ray.oz;

    s2tx = node.top[0] - ray.ox;
    s2ty = node.top[1] - ray.oy;
    s2tz = node.top[2] - ray.oz;

    switch(ray.dclass)
    {
        // MMM
        case 0:
        if (s2bx > 0.0f || s2by > 0.0f || s2bz > 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

        if (rx*s2by - ry*s2tx < 0.0f ||
            rx*s2ty - ry*s2bx > 0.0f ||
            rx*s2tz - rz*s2bx > 0.0f ||
            rx*s2bz - rz*s2tx < 0.0f ||
            ry*s2bz - rz*s2ty < 0.0f ||
            ry*s2tz - rz*s2by > 0.0f ) return false;
        break;

        // PMM
        case 1:
        if (s2tx < 0.0f || s2by > 0.0f || s2bz > 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

        if (rx*s2ty - ry*s2tx < 0.0f ||
            rx*s2by - ry*s2bx > 0.0f ||
            rx*s2bz - rz*s2bx > 0.0f ||
            rx*s2tz - rz*s2tx < 0.0f ||
            ry*s2bz - rz*s2ty < 0.0f ||
            ry*s2tz - rz*s2by > 0.0f) return false;
        break;

        // MPM
        case 2:
        if (s2bx > 0.0f || s2ty < 0.0f || s2bz > 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2by - ry*length > 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

        if (rx*s2by - ry*s2bx < 0.0f ||
            rx*s2ty - ry*s2tx > 0.0f ||
            rx*s2tz - rz*s2bx > 0.0f ||
            rx*s2bz - rz*s2tx < 0.0f ||
            ry*s2tz - rz*s2ty < 0.0f ||
            ry*s2bz - rz*s2by > 0.0f) return false;
        break;

        // PPM
        case 3:
        if (s2tx < 0.0f || s2ty < 0.0f || s2bz > 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2by - ry*length > 0.0f ||
            s2tz - rz*length < 0.0f) return false; // past length of ray

        if (rx*s2ty - ry*s2bx < 0.0f ||
            rx*s2by - ry*s2tx > 0.0f ||
            rx*s2bz - rz*s2bx > 0.0f ||
            rx*s2tz - rz*s2tx < 0.0f ||
            ry*s2tz - rz*s2ty < 0.0f ||
            ry*s2bz - rz*s2by > 0.0f) return false;
        break;

        // MMP
        case 4:
        if (s2bx > 0.0f || s2by > 0.0f || s2tz < 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

        if (rx*s2by - ry*s2tx < 0.0f ||
            rx*s2ty - ry*s2bx > 0.0f ||
            rx*s2tz - rz*s2tx > 0.0f ||
            rx*s2bz - rz*s2bx < 0.0f ||
            ry*s2bz - rz*s2by < 0.0f ||
            ry*s2tz - rz*s2ty > 0.0f) return false;
        break;

        // PMP
        case 5:
        if (s2tx < 0.0f || s2by > 0.0f || s2tz < 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2ty - ry*length < 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

        if (rx*s2ty - ry*s2tx < 0.0f ||
            rx*s2by - ry*s2bx > 0.0f ||
            rx*s2bz - rz*s2tx > 0.0f ||
            rx*s2tz - rz*s2bx < 0.0f ||
            ry*s2bz - rz*s2by < 0.0f ||
            ry*s2tz - rz*s2ty > 0.0f) return false;
        break;

        // MPP
        case 6:
        if (s2bx > 0.0f || s2ty < 0.0f || s2tz < 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2tx - rx*length < 0.0f ||
            s2by - ry*length > 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

        if (rx*s2by - ry*s2bx < 0.0f ||
            rx*s2ty - ry*s2tx > 0.0f ||
            rx*s2tz - rz*s2tx > 0.0f ||
            rx*s2bz - rz*s2bx < 0.0f ||
            ry*s2tz - rz*s2by < 0.0f ||
            ry*s2bz - rz*s2ty > 0.0f) return false;
        break;

        // PPP
        case 7:
        if (s2tx < 0.0f || s2ty < 0.0f || s2tz < 0.0f)
            return false; // AABB entirely in wrong octant wrt ray origin

        if (s2bx - rx*length > 0.0f ||
            s2by - ry*length > 0.0f ||
            s2bz - rz*length > 0.0f) return false; // past length of ray

        if (rx*s2ty - ry*s2bx < 0.0f ||
            rx*s2by - ry*s2tx > 0.0f ||
            rx*s2bz - rz*s2tx > 0.0f ||
            rx*s2tz - rz*s2bx < 0.0f ||
            ry*s2tz - rz*s2by < 0.0f ||
            ry*s2bz - rz*s2ty > 0.0f) return false;
        break;
    }
    // Didn't return false above, so we have a hit.
    return true;

}

__global__ void AABB_hit_plucker_kernel(const grace::Ray* rays,
                                        const grace::Node* nodes,
                                        const int N_rays,
                                        const int N_AABBs,
                                        unsigned int* ray_hits,
                                        unsigned int* box_hits)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        for (int i=0; i<N_AABBs; i++) {
            if (AABB_hit_plucker(rays[tid], nodes[i]))
            {
                ray_hits[tid]++;
                //atomicAdd(&(box_hits[i]), 1);
            }
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

    // Generate rays.  RNG in [-1, 1).
    random_float_functor rng(-1.0f, 1.0f);
    float x, y, z, N;
    for (int i=0; i<N_rays; i++) {
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
    float x0, x1, y0, y1, z0, z1;
    for (int i=0; i<N_AABBs; i++) {
        x0 = rng(3*N_rays + 6*i+0);
        x1 = rng(3*N_rays + 6*i+1);
        y0 = rng(3*N_rays + 6*i+2);
        y1 = rng(3*N_rays + 6*i+3);
        z0 = rng(3*N_rays + 6*i+4);
        z1 = rng(3*N_rays + 6*i+5);

        h_nodes[i].top[0] = max(x0, x1);
        h_nodes[i].top[1] = max(y0, y1);
        h_nodes[i].top[2] = max(z0, z1);

        h_nodes[i].bottom[0] = min(x0, x1);
        h_nodes[i].bottom[1] = min(y0, y1);
        h_nodes[i].bottom[2] = min(z0, z1);
    }

    thrust::device_vector<grace::Ray> d_rays = h_rays;
    thrust::device_vector<grace::Node> d_nodes = h_nodes;
    thrust::device_vector<unsigned int> d_ray_hits(N_rays);
    thrust::device_vector<unsigned int> d_box_hits(N_AABBs);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AABB_hit_eisemann_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_nodes.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()),
        thrust::raw_pointer_cast(d_box_hits.data()));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_eisemann_ray_hits = d_ray_hits;
    thrust::host_vector<unsigned int> h_eisemann_box_hits = d_box_hits;

    std::cout << N_rays << " rays tested against " << N_AABBs
              << " AABBs (ray slopes) in" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;


    // Perform plucker ray-box intersection tests on CPU.
    // thrust::fill(h_ray_hits.begin(), h_ray_hits.end(), 0u);
    // t = (double)clock() / CLOCKS_PER_SEC;
    // for (int i=0; i<N_rays; i++) {
    //     for (int j=0; j<N_AABBs; j++) {
    //         if (AABB_hit_plucker(h_rays[i], h_nodes[j]))
    //         {
    //             h_ray_hits[i]++;
    //             // h_box_hits[i]++;
    //         }
    //     }
    // }
    // t = (double)clock() / CLOCKS_PER_SEC - t;

    thrust::fill(d_ray_hits.begin(), d_ray_hits.end(), 0u);
    thrust::fill(d_box_hits.begin(), d_box_hits.end(), 0u);
    cudaEventRecord(start);
    AABB_hit_plucker_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_nodes.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()),
        thrust::raw_pointer_cast(d_box_hits.data()));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_plucker_ray_hits = d_ray_hits;
    thrust::host_vector<unsigned int> h_plucker_box_hits = d_box_hits;

    std::cout << N_rays << " rays tested against " << N_AABBs
              << " AABBs (plucker) in" << std::endl;
    // std::cout << "    CPU: " << t*1000. << " ms." << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;
    std::cout << std::endl;

    for (int i=0; i<N_rays; i++)
    {
        if (h_eisemann_ray_hits[i] != h_plucker_ray_hits[i])
        {
            std::cout << "Ray " << i << ": Eisemann (" << h_eisemann_ray_hits[i]
                      << ") != (" << h_plucker_ray_hits[i] << ") Plucker!"
                      << std::endl;
            std::cout << "ray.dclass = " << h_rays[i].dclass << std::endl;
            std::cout << "ray (ox, oy, oz): (" << h_rays[i].ox << ", "
                      << h_rays[i].oy << ", " << h_rays[i].oz << ")."
                      << std::endl;
            std::cout << "ray (dx, dy, dz): (" << h_rays[i].dx << ", "
                      << h_rays[i].dy << ", " << h_rays[i].dz << ")."
                      << std::endl;
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;

    for (int i=0; i<N_AABBs; i++)
    {
        if (h_eisemann_box_hits[i] != h_plucker_box_hits[i])
        {
            std::cout << "Box " << i << ": Eisemann (" << h_eisemann_box_hits[i]
                      << ") != (" << h_plucker_box_hits[i] << ") Plucker!"
                      << std::endl;
            std::cout << "AABB (bx, by, bz): (" << h_nodes[i].bottom[0] << ", "
                      << h_nodes[i].bottom[1] << ", " << h_nodes[i].bottom[2]
                      << ")" << std::endl;
            std::cout << "AABB (tx, ty, tz): (" << h_nodes[i].top[0] << ", "
                      << h_nodes[i].top[1] << ", " << h_nodes[i].top[2]
                      << ")" << std::endl;
            std::cout << std::endl;
        }
    }

}
