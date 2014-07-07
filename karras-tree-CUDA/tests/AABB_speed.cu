#include <sstream>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../nodes.h"
#include "../ray.h"
#include "../utils.cuh"
#include "../kernels/bintree_trace.cuh"
#include "../kernels/morton.cuh"
#include "../kernel_config.h"

__device__ __inline__ int   min_min   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin (float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax (float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin (float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax (float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d){ return fmax_fmax( fminf(a0,a1), fminf(b0,b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d)  { return fmin_fmin( fmaxf(a0,a1), fmaxf(b0,b1), fmax_fmin(c0, c1, d)); }

__device__ int AABB_hit_Aila_Laine(const float3 invd, const float4 ood,
                                   const float4 AABBx,
                                   const float4 AABBy,
                                   const float4 AABBz)
{
    float c0lox = AABBx.x * invd.x - ood.x;
    float c0hix = AABBx.y * invd.x - ood.x;
    float c0loy = AABBy.x * invd.y - ood.y;
    float c0hiy = AABBy.y * invd.y - ood.y;
    float c0loz = AABBz.x * invd.z - ood.z;
    float c0hiz = AABBz.y * invd.z - ood.z;
    float c1loz = AABBz.z * invd.z - ood.z;
    float c1hiz = AABBz.w * invd.z - ood.z;
    float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, 0);
    float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, ood.w);
    float c1lox = AABBx.z * invd.x - ood.x;
    float c1hix = AABBx.w * invd.x - ood.x;
    float c1loy = AABBy.z * invd.y - ood.y;
    float c1hiy = AABBy.w * invd.y - ood.y;
    float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 0);
    float c1max = spanEndKepler  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, ood.w);

    return (int)(c0max >= c0min) + (int)(c1max >= c1min);
}

__host__ __device__ bool AABB_hit_plucker(const grace::Ray& ray,
                                          const float bx,
                                          const float by,
                                          const float bz,
                                          const float tx,
                                          const float ty,
                                          const float tz)
{
    float rx = ray.dx;
    float ry = ray.dy;
    float rz = ray.dz;
    float length = ray.length;

    float s2bx, s2by, s2bz; // Vector from ray start to lower cell corner.
    float s2tx, s2ty, s2tz; // Vector from ray start to upper cell corner.

    s2bx = bx - ray.ox;
    s2by = by - ray.oy;
    s2bz = bz - ray.oz;

    s2tx = tx - ray.ox;
    s2ty = ty - ray.oy;
    s2tz = tz - ray.oz;

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

__global__ void AABB_hit_Aila_Laine_kernel(const grace::Ray* rays,
                                           const float4* AABBs,
                                           const int N_rays,
                                           const int N_AABBs,
                                           unsigned int* ray_hits)
{
    float4 AABBx, AABBy, AABBz, ood;
    grace::Ray ray;
    float3 invd;
    float ooeps = exp2f(-80.0f);
    unsigned int hit_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        ray = rays[tid];
        hit_count = 0;

        // Avoid div by zero.
        invd.x = 1.0f / (fabsf(ray.dx) > ooeps ? ray.dx : copysignf(ooeps, ray.dx));
        invd.y = 1.0f / (fabsf(ray.dy) > ooeps ? ray.dy : copysignf(ooeps, ray.dy));
        invd.z = 1.0f / (fabsf(ray.dz) > ooeps ? ray.dz : copysignf(ooeps, ray.dz));
        ood.x = ray.ox * invd.x;
        ood.y = ray.oy * invd.y;
        ood.z = ray.oz * invd.z;
        ood.w = ray.length;

        for (int i=0; i<N_AABBs/2; i++)
        {
            AABBx = AABBs[3*i + 0];
            AABBy = AABBs[3*i + 1];
            AABBz = AABBs[3*i + 2];
            hit_count += AABB_hit_Aila_Laine(invd, ood, AABBx, AABBy, AABBz);
        }

        ray_hits[tid] = hit_count;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void AABB_hit_eisemann_kernel(const grace::Ray* rays,
                                         const float4* AABBs,
                                         const int N_rays,
                                         const int N_AABBs,
                                         unsigned int* ray_hits)
{
    float4 AABBx, AABBy, AABBz;
    grace::Ray ray;
    grace::RaySlope slope;
    unsigned int hit_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        ray = rays[tid];
        slope = grace::ray_slope(ray);
        hit_count = 0;

        for (int i=0; i<N_AABBs/2; i++) {
            AABBx = AABBs[3*i + 0];
            AABBy = AABBs[3*i + 1];
            AABBz = AABBs[3*i + 2];
            if (grace::AABB_hit_eisemann(ray, slope,
                                         AABBx.x, AABBy.x, AABBz.x,
                                         AABBx.y, AABBy.y, AABBz.y))
                hit_count++;
            if (grace::AABB_hit_eisemann(ray, slope,
                                         AABBx.z, AABBy.z, AABBz.z,
                                         AABBx.w, AABBy.w, AABBz.w))
                hit_count++;
        }

        ray_hits[tid] = hit_count;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void AABB_hit_plucker_kernel(const grace::Ray* rays,
                                        const float4* AABBs,
                                        const int N_rays,
                                        const int N_AABBs,
                                        unsigned int* ray_hits)
{
    float4 AABBx, AABBy, AABBz;
    grace::Ray ray;
    unsigned int hit_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_rays)
    {
        ray = rays[tid];
        hit_count = 0;

        for (int i=0; i<N_AABBs/2; i++) {
            AABBx = AABBs[3*i + 0];
            AABBy = AABBs[3*i + 1];
            AABBz = AABBs[3*i + 2];
            if (AABB_hit_plucker(ray,
                                 AABBx.x, AABBy.x, AABBz.x,
                                 AABBx.y, AABBy.y, AABBz.y))
                hit_count++;
            if (AABB_hit_plucker(ray,
                                 AABBx.z, AABBy.z, AABBz.z,
                                 AABBx.w, AABBy.w, AABBz.w))
                hit_count++;
        }

        ray_hits[tid] = hit_count;

        tid += blockDim.x * gridDim.x;
    }
}


int main(int argc, char* argv[]) {

    /* Initialize run parameters. */

    unsigned int N_rays = 100000;
    unsigned int N_AABBs = 2*500;

    if (argc > 1)
        N_rays = (unsigned int) std::strtol(argv[1], NULL, 10);
    if (argc > 2)
        N_AABBs = 2*(unsigned int) std::strtol(argv[2], NULL, 10);

    std::cout << "Testing " << N_rays << " rays against "
              << N_AABBs << " AABBs." << std::endl;
    std::cout << std::endl;


    /* Generate the rays, emitted from (0, 0, 0) in a random direction.
     * NB: *Not* uniform random on surface of sphere.
     */

    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<float4> h_AABBs(3*(N_AABBs/2));
    thrust::host_vector<unsigned int> h_keys(N_rays);

    grace::random_float_functor rng(-1.0f, 1.0f);
    for (int i=0; i<N_rays; i++) {
        float x, y, z, N;

        x = rng(3*i + 0);
        y = rng(3*i + 1);
        z = rng(3*i + 2);

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

        // Floats must be in (0, 1) for morton_key().
        h_keys[i] = grace::morton_key((h_rays[i].dx+1)/2.f,
                                      (h_rays[i].dy+1)/2.f,
                                      (h_rays[i].dz+1)/2.f);
    }
    // Sort rays by Morton key.
    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    h_keys.clear();
    h_keys.shrink_to_fit();


    /* Generate the AABBs, with all points uniformly random in [-1, 1). */

    float bx, tx, by, ty, bz, tz;
    for (int i=0; i<N_AABBs/2; i++) {
        // ~ Left child AABB.
        bx = rng(3*N_rays + 12*i+0);
        ty = rng(3*N_rays + 12*i+1);
        by = rng(3*N_rays + 12*i+2);
        ty = rng(3*N_rays + 12*i+3);
        bz = rng(3*N_rays + 12*i+4);
        tz = rng(3*N_rays + 12*i+5);

        h_AABBs[3*i + 0].x = min(bx, tx);
        h_AABBs[3*i + 1].x = min(by, ty);
        h_AABBs[3*i + 2].x = min(bz, tz);

        h_AABBs[3*i + 0].y = max(bx, tx);
        h_AABBs[3*i + 1].y = max(by, ty);
        h_AABBs[3*i + 2].y = max(bz, tz);

        // ~ Right child AABB.
        bx = rng(3*N_rays + 12*i+6);
        ty = rng(3*N_rays + 12*i+7);
        by = rng(3*N_rays + 12*i+8);
        ty = rng(3*N_rays + 12*i+9);
        bz = rng(3*N_rays + 12*i+10);
        tz = rng(3*N_rays + 12*i+11);

        h_AABBs[3*i + 0].z = min(bx, tx);
        h_AABBs[3*i + 1].z = min(by, ty);
        h_AABBs[3*i + 2].z = min(bz, tz);

        h_AABBs[3*i + 0].w = max(bx, tx);
        h_AABBs[3*i + 1].w = max(by, ty);
        h_AABBs[3*i + 2].w = max(bz, tz);
    }

    thrust::device_vector<grace::Ray> d_rays = h_rays;
    thrust::device_vector<float4> d_AABBs = h_AABBs;
    thrust::device_vector<unsigned int> d_ray_hits(N_rays);
    thrust::host_vector<unsigned int> h_ray_hits(N_rays);


    /* Profile Aila and Laine. */

    // On GPU only.
    cudaEvent_t start, stop;
    float elapsed;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AABB_hit_Aila_Laine_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_AABBs.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()));
    cudaEventRecord(stop);
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_aila_laine_ray_hits = d_ray_hits;

    // Print results.
    std::cout << "Aila and Laine:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;


    /* Profile Eisemann. */

    // On GPU.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    AABB_hit_eisemann_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_AABBs.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()));
    cudaEventRecord(stop);
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_eisemann_ray_hits = d_ray_hits;

    // On CPU.
    double t = (double)clock() / CLOCKS_PER_SEC;
    for (int i=0; i<N_rays; i++) {
        grace::Ray ray = h_rays[i];
        grace::RaySlope slope = grace::ray_slope(ray);
        unsigned int hit_count = 0;

        for (int j=0; j<N_AABBs/2; j++) {
            float4 AABBx = h_AABBs[3*j + 0];
            float4 AABBy = h_AABBs[3*j + 1];
            float4 AABBz = h_AABBs[3*j + 2];
            if (AABB_hit_eisemann(ray, slope,
                                  AABBx.x, AABBy.x, AABBz.x,
                                  AABBx.y, AABBy.y, AABBz.y))
                hit_count++;
            if (AABB_hit_eisemann(ray, slope,
                                  AABBx.z, AABBy.z, AABBz.z,
                                  AABBx.w, AABBy.w, AABBz.w))
                hit_count++;
            h_ray_hits[i] = hit_count;
        }
    }
    t = (double)clock() / CLOCKS_PER_SEC - t;

    // Print results.
    std::cout << "Eisemann:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;
    std::cout << "    CPU: " << t*1000. << " ms." << std::endl;


    /* Profile Plucker. */

    // On GPU.
    cudaEventRecord(start);
    AABB_hit_plucker_kernel<<<48, 512>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_AABBs.data()),
        N_rays,
        N_AABBs,
        thrust::raw_pointer_cast(d_ray_hits.data()));
    cudaEventRecord(stop);
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    thrust::host_vector<unsigned int> h_plucker_ray_hits = d_ray_hits;

    // On CPU.
    thrust::fill(h_ray_hits.begin(), h_ray_hits.end(), 0u);
    t = (double)clock() / CLOCKS_PER_SEC;
    for (int i=0; i<N_rays; i++) {
        grace::Ray ray = h_rays[i];
        unsigned int hit_count = 0;

        for (int j=0; j<N_AABBs/2; j++) {
            float4 AABBx = h_AABBs[3*j + 0];
            float4 AABBy = h_AABBs[3*j + 1];
            float4 AABBz = h_AABBs[3*j + 2];
            if (AABB_hit_plucker(ray,
                                 AABBx.x, AABBy.x, AABBz.x,
                                 AABBx.y, AABBy.y, AABBz.y))
                hit_count++;
            if (AABB_hit_plucker(ray,
                                 AABBx.z, AABBy.z, AABBz.z,
                                 AABBx.w, AABBy.w, AABBz.w))
                hit_count++;
            h_ray_hits[i] = hit_count;
        }
    }
    t = (double)clock() / CLOCKS_PER_SEC - t;

    // Print results.
    std::cout << "Plucker:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;
    std::cout << "    CPU: " << t*1000. << " ms." << std::endl;
    std::cout << std::endl;


    /* Check Plucker and Eisemann intersection results match. */

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

    /* Check Aila and Plucker intersection results match. */

    for (int i=0; i<N_rays; i++)
    {
        if (h_aila_laine_ray_hits[i] != h_plucker_ray_hits[i])
        {
            std::cout << "Ray " << i << ": Aila (" << h_aila_laine_ray_hits[i]
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

    return 0;

}
