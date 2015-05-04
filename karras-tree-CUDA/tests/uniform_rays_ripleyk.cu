#include <math.h>

#include <curand.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../kernels/gen_rays.cuh"
#include "../ray.h"

int main(int argc, char* argv[]) {

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(5);

    size_t N_rays = 32 * 300; // = 9600
    if (argc > 1) {
        N_rays = 32 * (unsigned int) std::strtol(argv[1], NULL, 10);
    }

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::host_vector<grace::Ray> h_rays(N_rays);

    float ox, oy, oz;
    ox = oy = oz = 0;
    float length = 1;
    grace::uniform_random_rays(d_rays, ox, oy, oz, length);

    h_rays = d_rays;

    // Header of output table.
    std::cout << "Testing uniformity of " << N_rays << " generated ray "
              << "directions with Ripley's K function." << std::endl;
    std::cout << "L(r) - r = 0 when perfectly uniform on scale r." << std::endl;
    std::cout << std::endl;
    std::cout << "     r     |    L(r)   |  L(r) - r  " << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    // Below required for calculating K function.
    double inv_N_rays = 1. / N_rays;
    double prefactor = 4 * inv_N_rays;

    // Loop through various distance scales.
    const size_t N_scales = 10;
    float Rs[N_scales] = {0.005, 0.01, 0.02, 0.03, 0.05,
                          0.1, 0.2, 0.5, 1.0, sqrt(2)};
    for (unsigned int ir=0; ir<N_scales; ir++)
    {
        float r = Rs[ir];
        double Lr;
        unsigned int N_within_tot = 0;
        // Loop through each particle.
        for (unsigned int i=0; i<N_rays; i++)
        {
            float dxi, dyi, dzi;
            dxi = h_rays[i].dx;
            dyi = h_rays[i].dy;
            dzi = h_rays[i].dz;

            for (unsigned int j=0; j<N_rays; j++)
            {
                // Do not include a point as part of its own count.
                if (i == j)
                    continue;

                float dxj, dyj, dzj;
                dxj = h_rays[j].dx;
                dyj = h_rays[j].dy;
                dzj = h_rays[j].dz;

                if ( (dxi-dxj)*(dxi-dxj) +
                     (dyi-dyj)*(dyi-dyj) +
                     (dzi-dzj)*(dzi-dzj) < r*r)
                {
                    N_within_tot++;
                }
            }
        }
        double N_within_avg = N_within_tot * inv_N_rays;
        // L(r) = sqrt[K(r)/pi], where K(r) is Ripley's K function.
        Lr = sqrt(prefactor * N_within_avg);
        std::cout << "  " << r << "  |  " << Lr << "  |  ";
        std::cout.width(8);
        std::cout << Lr - r << std::endl;
    }
    return 0;
}
