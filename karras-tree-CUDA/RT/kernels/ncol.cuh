#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "../../particle.h"

namespace grace {

namespace gpu {

__global__ void fill_ncols_HI(const particle_ion* particles,
                              const Float* integrals,
                              const unsigned int* indices,
                              const Float N_H,
                              Float* Ncols,
                              size_t N_intersections)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int particle_index;

    while (tid < N_intersections)
    {
        particle_index = indices[tid];
        Ncols[tid] = N_H * particles[particle_index].x_HI * integrals[tid];
        tid += blockDim.x * gridDim.x
    }
}

__global__ void fill_ncols_HeI(const particle_ion* particles,
                               const Float* integrals,
                               const unsigned int* indices,
                               const Float N_He,
                               Float* Ncols,
                               size_t N_intersections)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_intersections)
    {
        Ncols[tid] = N_He * particles[tid].x_HeI * integrals[tid];
        tid += blockDim.x * gridDim.x
    }
}

__global__ void fill_ncols_HeII(const particle_ion* particles,
                                const Float* integrals,
                                const unsigned int* indices,
                                const Float N_He,
                                Float* Ncols,
                                size_t N_intersections)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_intersections)
    {
        Ncols[tid] = N_He * particles[tid].x_HeII * integrals[tid];
        tid += blockDim.x * gridDim.x
    }
}

} // namespace gpu


template <typename Float>
void cum_ncol_HI(const thrust::device_vector<particle_ion>& d_particles,
                 const thrust::device_vector<Float> d_integrals,
                 const thrust::device_vector<unsigned int>& d_indices,
                 const thrust::device_vector<unsigned int>& d_ray_segments,
                 const Float N_H_per_particle,
                 thrust::device_vector<Float>& d_Ncol)
{
    // Initialize d_Ncol with the column densities through each particle
    // (not cumulative!)
    gpu::fill_ncols_HI<<<48,512>>>(
        thrust::raw_pointer_cast(d_particles.data()),
        thrust::raw_pointer_cast(d_integrals.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        N_H_per_particle,
        thrust::raw_pointer_cast(d_Ncol.data()),
        d_indices.size())

    // Sum such that d_Ncol[i] = cumulative Ncol *up to* (not including) the
    // ith intersected particle.
    thrust::exclusive_scan_by_key(d_ray_segments.begin(), d_ray_segments.end(),
                                  d_Ncol.begin(), d_Ncol.begin(), 0.f);
}

template <typename Float>
void cum_ncol_HeI(const thrust::device_vector<particle_ion>& d_particles,
                  const thrust::device_vector<Float> d_integrals,
                  const thrust::device_vector<unsigned int>& d_indices,
                  const thrust::device_vector<unsigned int>& d_ray_segments,
                  const Float N_He_per_particle,
                  thrust::device_vector<Float>& d_Ncol)
{
    gpu::fill_ncols_HeI<<<48,512>>>(
        thrust::raw_pointer_cast(d_particles.data()),
        thrust::raw_pointer_cast(d_integrals.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        N_He_per_particle,
        thrust::raw_pointer_cast(d_Ncol.data()),
        d_indices.size())

    thrust::exclusive_scan_by_key(d_ray_segments.begin(), d_ray_segments.end(),
                                  d_Ncol.begin(), d_Ncol.begin(), 0.f);
}

template <typename Float>
void cum_ncol_HeII(const thrust::device_vector<particle_ion>& d_particles,
                   const thrust::device_vector<Float> d_integrals,
                   const thrust::device_vector<unsigned int>& d_indices,
                   const thrust::device_vector<unsigned int>& d_ray_segments,
                   const Float N_He_per_particle,
                   thrust::device_vector<Float>& d_Ncol)
{
    gpu::fill_ncols_HeII<<<48,512>>>(
        thrust::raw_pointer_cast(d_particles.data()),
        thrust::raw_pointer_cast(d_integrals.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        N_He_per_particle,
        thrust::raw_pointer_cast(d_Ncol.data()),
        d_indices.size())

    thrust::exclusive_scan_by_key(d_ray_segments.begin(), d_ray_segments.end(),
                                  d_Ncol.begin(), d_Ncol.begin(), 0.f);
}

} // namespace grace
