#include <fstream>
#include <iomanip>
#include <math.h>

#include <curand.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../error.h"
#include "../ray.h"
#include "../kernels/gen_rays.cuh"
#include "../util/meta.h"

double PI = 3.141592653589793;

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

template <typename Real2>
struct real2_add : public thrust::binary_function<Real2, Real2, Real2>
{
    __host__ __device__
    Real2 operator()(Real2 lhs, Real2 rhs) const
    {
        Real2 res;
        res.x = lhs.x + rhs.x;
        res.y = lhs.y + rhs.y;
        return res;
    }
};

template <typename Real2>
__global__ void compute_ij_pairs(grace::Ray* rays,
                                 Real2* psi_sin_ijs,
                                 size_t i,
                                 size_t j_start,
                                 size_t N)
{
    typedef typename grace::Real2ToRealMapper<Real2>::type Real;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        Real xi = rays[i].dx;
        Real yi = rays[i].dy;
        Real zi = rays[i].dz;

        Real xj = rays[j_start + tid].dx;
        Real yj = rays[j_start + tid].dy;
        Real zj = rays[j_start + tid].dz;

        // Ray direction vectors are already normalized.
        Real cosij = xi*xj + yi*yj + zi*zj;
        cosij = max(-1., min(1., cosij));

        // We require an angle in [0, pi], which acos provides.
        Real psiij = acos(cosij);
        Real sinij = sqrt(1. - cosij*cosij);

        psi_sin_ijs[tid].x = psiij;
        psi_sin_ijs[tid].y = sinij;
    }

    return;
}

int main(int argc, char* argv[]) {
    // Precision for statistical calculations.
    // Significant memory requirements, so float may be preferable depending
    // on the device.
    typedef float Real;
    typedef float2 Real2;

    std::cout.precision(6);
    std::cout.fill('0');
    std::cout.setf(std::ios::fixed);

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

    std::cout << "For details of these tests, see uniform_rays_stats.md"
              << std::endl;
    std::cout << std::endl;

    thrust::device_vector<grace::Ray> d_rays(N_rays);

    float ox, oy, oz;
    ox = oy = oz = 0;
    float length = 1;
    grace::uniform_random_rays(d_rays, ox, oy, oz, length);

    grace::Ray zero_ray;
    zero_ray.dx = zero_ray.dy = zero_ray.dz = 0.f;
    grace::Ray raydir_sum = thrust::reduce(d_rays.begin(), d_rays.end(),
                                           zero_ray, raydir_add());
    Real R2 = raydir_sum.dx * raydir_sum.dx +
              raydir_sum.dy * raydir_sum.dy +
              raydir_sum.dz * raydir_sum.dz;
    Real z = 3. * R2 / N_rays;

    std::cout << "z is:  " << z
              << "  P(z > 7.815) = 0.05" << std::endl;
    std::cout << std::endl;

    // This is broken into iterations to save on memory. It is fast enough
    // despite incurring substantial overhead via Thrust/CUDA calls.
    thrust::host_vector<Real2> ij_sums(N_rays - 1);
    for (size_t i = 0; i < N_rays - 1; i++) {
        size_t j_start = i + 1;
        size_t N_pairs = N_rays - j_start;
        thrust::device_vector<Real2> psi_sinpsi_ij(N_pairs);

        size_t num_blocks = (N_pairs + 128 - 1) / 128;
        compute_ij_pairs<<<num_blocks, 128>>>(
            thrust::raw_pointer_cast(d_rays.data()),
            thrust::raw_pointer_cast(psi_sinpsi_ij.data()),
            i,
            j_start,
            N_pairs);
        GRACE_KERNEL_CHECK();

        Real2 zero; zero.x = zero.y = 0.0;
        Real2 sums = thrust::reduce(psi_sinpsi_ij.begin(), psi_sinpsi_ij.end(),
                                    zero, real2_add<Real2>());
        ij_sums[i] = sums;
    }

    // We compute this on the host so we can specify the order of summation.
    // ij_sums will be a (mostly-)monotonically-decreasing vector of positive
    // numbers.
    // We sum them from the last element to the first, in order to reduce the
    // difference in magnitude between the two numbers being summed and thus
    // increase the accuracy of the final value.
    Real psi_ij_sum = 0.0;
    Real sinpsi_ij_sum = 0.0;
    for (int i = ij_sums.size() - 1; i >= 0; --i)
    {
        Real2 ij_sum = ij_sums[i];
        psi_ij_sum += ij_sum.x;
        sinpsi_ij_sum += ij_sum.y;
    }

    Real coeff = 4. / (N_rays * PI);
    Real An = N_rays - coeff * psi_ij_sum;
    Real Gn = N_rays / 2. - coeff * sinpsi_ij_sum;

    std::cout << "An is: " << An
              << "  P(An > 1.4136) = 0.20" << std::endl
              << "       " << "          "
              << "P(An > 2.2073) = 0.05" << std::endl
              << std::endl;
    std::cout << "Gn is: " << Gn
              << "  P(Gn > 0.64643) = 0.20" << std::endl
              << "       " << "          "
              << "P(Gn > 0.88384) = 0.05" << std::endl
              << std::endl;
    std::cout << "Fn is: " << An + Gn
              << "  P(Fn > 1.9478) = 0.20" << std::endl
              << "       " << "          "
              << "P(Fn > 2.7477) = 0.05" << std::endl
              << std::endl;

    if (save_out) {
        thrust::host_vector<grace::Ray> h_rays = d_rays;

        std::ofstream outfile;
        outfile.setf(std::ios::fixed);
        outfile.precision(6);
        outfile.open("outdata/ray_dirs.txt");
        for (unsigned int i = 0; i < N_rays; i++) {
            grace::Ray ray = h_rays[i];
            outfile << ray.dx << ", " << ray.dy << ", " << ray.dz << std::endl;
        }
        outfile.close();
    }

    return EXIT_SUCCESS;
}
