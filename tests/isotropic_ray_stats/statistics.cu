#include "statistics.cuh"
#include "stats_math.cuh"

#include "grace/error.h"
#include "grace/ray.h"
#include "grace/vector.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <math.h>

// Forward declared. These are not in the header since they are not intended for
// use outwith this file.
__global__ void compute_ij_pairs(const grace::Ray*, double2*,
                                 const size_t, const size_t, const size_t);
void ij_pair_sums(const thrust::device_vector<grace::Ray>&,
                  thrust::host_vector<double2>&);


float resultant_length_squared(const thrust::device_vector<grace::Ray>& rays)
{
    grace::Ray zero_ray;
    zero_ray.dx = zero_ray.dy = zero_ray.dz = 0.f;

    grace::Ray raydir_sum = thrust::reduce(rays.begin(), rays.end(),
                                           zero_ray, raydir_add());

    return (double)raydir_sum.dx * raydir_sum.dx +
           (double)raydir_sum.dy * raydir_sum.dy +
           (double)raydir_sum.dz * raydir_sum.dz;
}

void An_Gn_statistics(const thrust::device_vector<grace::Ray>& rays,
                      double* An, double* Gn)
{
    const size_t N_rays = rays.size();

    // For each pair in ij_sums, pair.x is a sum of psi_ij angles, and pair.y
    // is a sum of sin(psi_ij) values.
    thrust::host_vector<double2> ij_sums(N_rays - 1);
    ij_pair_sums(rays, ij_sums);

    // We compute this on the host so we can specify the order of summation.
    // ij_sums will be a (mostly-)monotonically-decreasing and all positive.
    // We sum them from the last element to the first in order to reduce the
    // difference in magnitude between any two numbers being added, and thus
    // increase the accuracy of the final value.
    double psi_ij_sum = 0.0;
    double sinpsi_ij_sum = 0.0;
    for (int i = ij_sums.size() - 1; i >= 0; --i)
    {
        double2 ij_sum = ij_sums[i];

        psi_ij_sum    += ij_sum.x;
        sinpsi_ij_sum += ij_sum.y;
    }

    double coeff = 4.0 / (N_rays * PI);
    *An = N_rays - coeff * psi_ij_sum;
    *Gn = N_rays / 2.0 - coeff * sinpsi_ij_sum;
}


__global__ void compute_ij_pairs(const grace::Ray* rays,
                                 double2* psi_sin_ijs,
                                 const size_t i,
                                 const size_t j_start,
                                 const size_t N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // Rays always contain single-precision variables.
        grace::Vector<3, float> pi;
        pi.x = rays[i].dx;
        pi.y = rays[i].dy;
        pi.z = rays[i].dz;

        grace::Vector<3, float> pj;
        pj.x = rays[j_start + tid].dx;
        pj.y = rays[j_start + tid].dy;
        pj.z = rays[j_start + tid].dz;

        // Ray direction vectors are already normalized.
        double psiij = angular_separation(pi, pj);
        double sinij = sin(psiij);

        psi_sin_ijs[tid].x = psiij;
        psi_sin_ijs[tid].y = sinij;
    }

    return;
}

void ij_pair_sums(const thrust::device_vector<grace::Ray>& rays,
                  thrust::host_vector<double2>& ij_sums)
{
    const size_t N_rays = rays.size();
    double2 zero = make_double2(0.0, 0.0);

    // This is broken into iterations to save on memory. It is fast enough
    // despite incurring substantial overhead via Thrust/CUDA calls.
    thrust::device_vector<double2> psi_sinpsi_ij(N_rays - 1);
    for (size_t i = 0; i < N_rays - 1; ++i)
    {
        size_t j_start = i + 1;
        size_t N_pairs = N_rays - j_start;

        size_t num_blocks = (N_pairs + 127) / 128;
        compute_ij_pairs<<<num_blocks, 128>>>(
            thrust::raw_pointer_cast(rays.data()),
            thrust::raw_pointer_cast(psi_sinpsi_ij.data()),
            i,
            j_start,
            N_pairs);
        GRACE_KERNEL_CHECK();

        double2 sums = thrust::reduce(psi_sinpsi_ij.begin(),
                                      psi_sinpsi_ij.begin() + N_pairs,
                                      zero, real2_add<double2>());
        ij_sums[i] = sums;
    }
}
