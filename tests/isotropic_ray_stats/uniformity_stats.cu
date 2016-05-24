// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "statistics.cuh"

#include "grace/cuda/gen_rays.cuh"
#include "grace/ray.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>

int main(int argc, char* argv[]) {
    std::cout.precision(6);
    std::cout.setf(std::ios::fixed);

    size_t N_rays = 32 * 300; // = 9600
    bool save_out = false;
    if (argc > 1) {
        N_rays = 32 * (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
      save_out = (std::string(argv[2]) == "save") ? true : false;
    }

    std::cout << "For details of these tests, see isotropic_stats.md"
              << std::endl
              << std::endl;

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    grace::uniform_random_rays(d_rays, 0.f, 0.f, 0.f, 1.f);

    // Ray directions are always floats.
    float R2 = resultant_length_squared(d_rays);
    float z = 3.f * R2 / N_rays;

    std::cout << "z:  " << std::setw(8) << z
              << "  P(z > 7.815) = 0.05" << std::endl
              << std::endl;

    double An, Gn;
    An_Gn_statistics(d_rays, &An, &Gn);
    double Fn = An + Gn;

    std::cout << "An: " << std::setw(8) << An
              << "  P(An > 1.4136) = 0.20" << std::endl
              << "    " << std::setw(8) << " "
              << "  P(An > 2.2073) = 0.05" << std::endl
              << std::endl;
    std::cout << "Gn: " << std::setw(8) << Gn
              << "  P(Gn > 0.64643) = 0.20" << std::endl
              << "    " << std::setw(8) << " "
              << "  P(Gn > 0.88384) = 0.05" << std::endl
              << std::endl;
    std::cout << "Fn: " << std::setw(8) << An + Gn
              << "  P(Fn > 1.9478) = 0.20" << std::endl
              << "    " << std::setw(8) << " "
              << "  P(Fn > 2.7477) = 0.05" << std::endl
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

    return (z <= 7.815 && Fn <= 1.9478) ? EXIT_SUCCESS : EXIT_FAILURE;
}
