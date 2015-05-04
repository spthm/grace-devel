#include <fstream>
#include <iomanip>
#include <math.h>

#include <curand.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../kernels/gen_rays.cuh"
#include "../ray.h"

template <typename Real>
__global__ void fill_XYZs(grace::Ray* rays,
                          Real* Xs,
                          Real* Ys,
                          Real* Zs,
                          size_t N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        grace::Ray ray = rays[tid];
        Xs[tid] = ray.dx;
        Ys[tid] = ray.dy;
        Zs[tid] = ray.dz;
    }

    return;
}

template <typename Real>
__global__ void compute_ij_pairs(Real* Xs,
                                 Real* Ys,
                                 Real* Zs,
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
        Real xi = Xs[i];
        Real yi = Ys[i];
        Real zi = Zs[i];

        Real xj = Xs[j_start + tid];
        Real yj = Ys[j_start + tid];
        Real zj = Zs[j_start + tid];

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

int main(int argc, char* argv[]) {

    double PI = 3.14159265358979323846;
    // Precision for statistical calculations.
    // Significant memory requirements, so float may be preferable depending
    // on the device.
    typedef double Real;

    size_t N_rays = 10000;
    bool save_out = false;
    if (argc > 1) {
        N_rays = (size_t) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        if (strcmp("save", argv[2]) == 0) {
            save_out = true;
        }
    }

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::host_vector<grace::Ray> h_rays(N_rays);

    float ox, oy, oz;
    ox = oy = oz = 0;
    float length = 1;
    grace::uniform_random_rays(d_rays, ox, oy, oz, length);

    h_rays = d_rays;

    std::ofstream outfile;
    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(6);

    thrust::device_vector<Real> d_X(N_rays);
    thrust::device_vector<Real> d_Y(N_rays);
    thrust::device_vector<Real> d_Z(N_rays);

    int num_blocks = (N_rays + 128 - 1) / 128;
    fill_XYZs<<<num_blocks, 128>>>(
        thrust::raw_pointer_cast(d_rays.data()),
        thrust::raw_pointer_cast(d_X.data()),
        thrust::raw_pointer_cast(d_Y.data()),
        thrust::raw_pointer_cast(d_Z.data()),
        N_rays);

    Real X_sum = thrust::reduce(d_X.begin(), d_X.end(), 0.);
    Real Y_sum = thrust::reduce(d_Y.begin(), d_Y.end(), 0.);
    Real Z_sum = thrust::reduce(d_Z.begin(), d_Z.end(), 0.);

    std::cout << "(X sum: " << X_sum << ")" << std::endl;
    std::cout << "(Y sum: " << Y_sum << ")" << std::endl;
    std::cout << "(Z sum: " << Z_sum << ")" << std::endl;
    std::cout << std::endl;

    Real R2 = X_sum*X_sum + Y_sum*Y_sum + Z_sum*Z_sum;
    Real W = 3. * R2 / N_rays;
    Real W_alt = (1. - (1. / (2.*N_rays)))*W + (1. / (10.*N_rays)) * W*W;

    std::cout << "W is:  " << W << std::endl;
    std::cout << "W* is: " << W_alt << std::endl;
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
            thrust::raw_pointer_cast(d_X.data()),
            thrust::raw_pointer_cast(d_Y.data()),
            thrust::raw_pointer_cast(d_Z.data()),
            thrust::raw_pointer_cast(cospsi_ij.data()),
            thrust::raw_pointer_cast(psi_ij.data()),
            thrust::raw_pointer_cast(sinpsi_ij.data()),
            i,
            j_start,
            N_pairs,
            write_offset);

        write_offset += N_pairs;
    }

    double psi_ij_sum = thrust::reduce(psi_ij.begin(), psi_ij.end(), 0.);
    double sinpsi_ij_sum = thrust::reduce(sinpsi_ij.begin(), sinpsi_ij.end(), 0.);
    double coeff = 4. / (N_rays * PI);
    double An = N_rays - coeff * psi_ij_sum;
    double Gn = N_rays / 2. - coeff * sinpsi_ij_sum;

    std::cout << "((4/Npi)*psi_ij.sum() is:    " << coeff * psi_ij_sum << ")"
              << std::endl;
    std::cout << "((4/Npi)*sinpsi_ij.sum() is: " << coeff * sinpsi_ij_sum << ")"
              << std::endl;
    std::cout << std::endl;

    std::cout << "An is: " << An << std::endl;
    std::cout << "Gn is: " << Gn << std::endl;
    std::cout << "Fn is: " << An + Gn << std::endl;

    if (save_out) {
        outfile.open("outdata/ray_dirs.txt");
        for (unsigned int i = 0; i < N_rays; i++) {
            grace::Ray ray = h_rays[i];
            outfile << ray.dx << ", " << ray.dy << ", " << ray.dz << std::endl;
        }
        outfile.close();
    }

    return 0;
}
