// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "stats_math.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ray.h"
#include "kernels/gen_rays.cuh"

#include <cmath>
#include <iomanip>

const size_t N_scales = 12;
float Rs[N_scales] = { 0.005,
                       0.01,
                       0.02,
                       0.03,
                       0.05,
                       0.1,
                       0.2,
                       0.5,
                       0.75,
                       1.0,
                       1.25,
                       PI / 2.0
                     };

int main(int argc, char* argv[])
{
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);

    size_t N_rays = 32 * 300; // = 9600
    if (argc > 1) {
        N_rays = 32 * (size_t)std::strtol(argv[1], NULL, 10);
    }

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::host_vector<grace::Ray> h_rays(N_rays);

    float3 O = make_float3(0.f, 0.f, 0.f);
    float length = 1;
    grace::uniform_random_rays(d_rays, O.x, O.y, O.z, length);

    h_rays = d_rays;

    // Header of output table.
    std::cout << "Testing uniformity of " << N_rays << " generated ray "
              << "directions with Ripley's K(r) function." << std::endl
              << "r is the distance scale: 1/2 arc length of a spherical cap "
              << "on the unit sphere." << std::endl
              << "CSR(r) is the value of K(r) under Complete Spatial "
              << "Randomness." << std::endl
              << std::endl
              << "      r     |    K(r)    | K(r) - CSR(r)" << std::endl
              << "----------------------------------------" << std::endl;

    #pragma omp parallel for schedule(static,1) ordered
    for (int s = 0; s < N_scales; ++s)
    {
        float r = Rs[s];
        int N_within = 0;

        for (int i = 0; i < N_rays; ++i)
        {
            float3 p;
            p.x = h_rays[i].dx;
            p.y = h_rays[i].dy;
            p.z = h_rays[i].dz;

            for (int j = 0; j < N_rays; ++j)
            {
                // Do not include a direction as part of its own count.
                if (i == j)
                    continue;

                float3 q;
                q.x = h_rays[j].dx;
                q.y = h_rays[j].dy;
                q.z = h_rays[j].dz;

                if (great_circle_distance(p, q) < r) {
                    ++N_within;
                }
            }
        }

        double CSR = 2 * PI * (1 - std::cos(r));
        double K = (4 * PI / (N_rays * (N_rays - 1))) * N_within;

        #pragma omp ordered
        {
            std::cout << "  " << std::setw(8) << r << "  |"
                      << "  " << std::setw(8) << K << "  |"
                      << "  " << std::setw(9) << K - CSR
                      << std::endl;
        }
    }
    return EXIT_SUCCESS;
}
