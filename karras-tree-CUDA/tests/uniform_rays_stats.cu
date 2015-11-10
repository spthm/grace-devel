#include <fstream>
#include <iomanip>
#include <math.h>

#include <curand.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../error.h"
#include "../kernels/gen_rays.cuh"
#include "../ray.h"

struct raydir_add : public thrust::binary_function<grace::Ray, grace::Ray, grace::Ray>
{
    __host__ __device__
    grace::Ray operator()(grace::Ray lhs, grace::Ray rhs) const
    {
        grace::Ray ray;
        ray.dx = lhs.dx + rhs.dx;
        ray.dy = lhs.dy + rhs.dy;
        ray.dz = lhs.dz + rhs.dz;
        return ray;
    }
};

template <typename Real>
__global__ void compute_ij_pairs(grace::Ray* rays,
                                 Real* cos_ijs,
                                 Real* psi_ijs,
                                 Real* sin_ijs,
                                 size_t i,
                                 size_t j_start,
                                 size_t N,
                                 size_t offset)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        Real xi = rays[i].dx;
        Real yi = rays[i].dy;
        Real zi = rays[i].dz;

        Real xj = rays[j_start + tid].dx;
        Real yj = rays[j_start + tid].dy;
        Real zj = rays[j_start + tid].dz;

        Real cosij = xi*xj + yi*yj + zi*zj;
        cosij = max(-1., min(1., cosij));

        Real psiij = acos(cosij);

        Real sinij = sqrt(1. - cosij*cosij);

        cos_ijs[offset + tid] = cosij;
        psi_ijs[offset + tid] = psiij;
        sin_ijs[offset + tid] = sinij;
    }

    return;
}

/*
 * For a desciption of the below statistical tests, see e.g.
 * "Statistical Analysis of Spherical Data" (1987) by N. I. Fisher, T. Lewis and
 * B. J. J. Embleton,  Cambridge University Press, Online ISBN:9780511623059.
 * Chapter 5.3 covers 'R^2 and W statistics, while 5.6 covers An, Gn and Fn
 * statistics.
 */

int main(int argc, char* argv[]) {

    double PI = 3.14159265358979323846;
    // Precision for statistical calculations.
    // Significant memory requirements, so float may be preferable depending
    // on the device.
    typedef float Real;

    size_t N_rays = 32 * 300; // = 9600
    bool save_out = false;
    if (argc > 1) {
        N_rays = 32 * (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        if (strcmp("save", argv[2]) == 0) {
            save_out = true;
        }
    }

    thrust::device_vector<grace::Ray> d_rays(N_rays);

    float ox, oy, oz;
    ox = oy = oz = 0;
    float length = 1;
    grace::uniform_random_rays(d_rays, ox, oy, oz, length);

    std::ofstream outfile;
    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(6);

    grace::Ray zero_ray;
    zero_ray.dx = zero_ray.dy = zero_ray.dx = 0.f;
    grace::Ray raydir_sum = thrust::reduce(d_rays.begin(), d_rays.end(),
                                           zero_ray, raydir_add());

    std::cout << "(X sum: " << raydir_sum.dx << ")" << std::endl;
    std::cout << "(Y sum: " << raydir_sum.dy << ")" << std::endl;
    std::cout << "(Z sum: " << raydir_sum.dz << ")" << std::endl;
    std::cout << std::endl;

    Real R2 = raydir_sum.dx * raydir_sum.dx +
              raydir_sum.dy * raydir_sum.dy +
              raydir_sum.dz * raydir_sum.dz;
    Real W = 3. * R2 / N_rays;
    Real W_alt = (1. - (1. / (2.*N_rays)))*W + (1. / (10.*N_rays)) * W*W;

    std::cout << "W is:  " << W << std::endl;
    std::cout << "W* is: " << W_alt << std::endl;
    std::cout << "W < (Chi_3)^2 (0.05) = 7.815 -> it is reasonable to assume"
              << std::endl
              << "directions are isotropic against a unimodal alternative"
              << std::endl
              << "(there is no evidence at the 5% level to support the unimodal"
              << std::endl
              << "alternative)."
              << std::endl;
    std::cout << std::endl;

    // SUM(i,j = 1 TO i,j = N; i != j) = (1/2)n(n-1), and n(n-1) is always even.
    size_t N_ij_paris = (N_rays * (N_rays - 1)) / 2;
    thrust::device_vector<Real> cospsi_ij(N_ij_paris);
    thrust::device_vector<Real> psi_ij(N_ij_paris);
    thrust::device_vector<Real> sinpsi_ij(N_ij_paris);

    size_t write_offset = 0;
    for (size_t i = 0; i < N_rays - 1; i++) {
        size_t j_start = i + 1;
        size_t N_pairs = N_rays - j_start;

        size_t num_blocks = (N_pairs + 128 - 1) / 128;
        compute_ij_pairs<<<num_blocks, 128>>>(
            thrust::raw_pointer_cast(d_rays.data()),
            thrust::raw_pointer_cast(cospsi_ij.data()),
            thrust::raw_pointer_cast(psi_ij.data()),
            thrust::raw_pointer_cast(sinpsi_ij.data()),
            i,
            j_start,
            N_pairs,
            write_offset);
        GRACE_KERNEL_CHECK();

        write_offset += N_pairs;
    }

    double psi_ij_sum = thrust::reduce(psi_ij.begin(), psi_ij.end(), 0.);
    double sinpsi_ij_sum = thrust::reduce(sinpsi_ij.begin(), sinpsi_ij.end(), 0.);
    double coeff = 4. / (N_rays * PI);
    double An = N_rays - coeff * psi_ij_sum;
    double Gn = N_rays / 2. - coeff * sinpsi_ij_sum;

    std::cout << "An is: " << An << std::endl;
    std::cout << "Gn is: " << Gn << std::endl;
    std::cout << "Fn is: " << An + Gn << std::endl;
    std::cout << std::endl;
    std::cout << "An is Beran's statistic, testing for uniformity against"
              << std::endl
              << "alternative models which are not symmetric wrt. the sphere's"
              << std::endl
              << "centre." << std::endl;
    std::cout << "An < 1.413 -> no evidence to reject the isotropic model at"
              << std::endl
              << "the 20% level."
              << std::endl;
    std::cout << "Gn is Gine's statistic, testing for uniformity against"
              << std::endl
              << "alternative models which are symmetric wrt. the sphere's"
              << std::endl
              << "centre." << std::endl;
    std::cout << "Gn < 0.646 -> no evidence to reject the isotropic model at"
              << std::endl
              << "the 20% level."
              << std::endl;
    std::cout << "Fn tests for uniformity against all alternative models."
              << std::endl;
    std::cout << "Fn < 1.948 -> no evidence to reject the isotropic model at"
              << std::endl
              << "the 20% level."
              << std::endl;

    if (save_out) {
        thrust::host_vector<grace::Ray> h_rays = d_rays;
        outfile.open("outdata/ray_dirs.txt");
        for (unsigned int i = 0; i < N_rays; i++) {
            grace::Ray ray = h_rays[i];
            outfile << ray.dx << ", " << ray.dy << ", " << ray.dz << std::endl;
        }
        outfile.close();
    }

    return 0;
}
