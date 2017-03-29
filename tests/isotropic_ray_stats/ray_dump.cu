// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/generate_rays.cuh"
#include "grace/cuda/prngstates.cuh"
#include "grace/ray.h"
#include "grace/vector.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iomanip>
#include <fstream>
#include <sstream>

template <typename T>
std::string number_string(T n)
{
    std::ostringstream ss;
    ss << n;
    return ss.str();
}

int main(int argc, char* argv[])
{
    size_t N_rays = 32 * 300; // = 9600
    size_t N_samples = 100;
    unsigned long long seed = 123456789;
    std::string out_dir = "ray_dumps/";
    if (argc > 1) {
        N_rays = 32 * (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_samples = (size_t)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        seed = (unsigned long long)std::strtol(argv[3], NULL, 10);
    }
    if (argc > 4) {
        out_dir = std::string(argv[4]) + std::string("/");
    }

    grace::Vector<3, float> origin(0.f, 0.f, 0.f);
    float length = 1.0;

    std::cout << "Dumping " << N_samples << " samples of " << N_rays
              << " rays centred at (" << origin.x << ", " << origin.y << ", "
              << origin.z << ")." << std::endl
              << "Seed value: " << seed << std::endl
              << "Output directory: " << out_dir << std::endl;

    grace::PrngStates rng_states(seed);
    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::host_vector<grace::Ray> h_rays(N_rays);

    std::ofstream outfile;
    outfile.precision(6);
    for (size_t i = 0; i < N_samples; ++i)
    {
        grace::uniform_random_rays(grace::Vector<3, float>(), length,
                                   rng_states, d_rays);
        h_rays = d_rays;

        std::string ofname = std::string("sample_") + number_string(i);
        outfile.open((out_dir + ofname).c_str());
        for (size_t r = 0; r < N_rays; ++r)
        {
            grace::Ray ray = h_rays[r];
            outfile << "ox:" << ray.ox << "oy:" << ray.oy << "oz:" << ray.oz
                    << "dx:" << ray.dx << "dy:" << ray.dy << "dz:" << ray.dz
                    << std::endl;
        }
        outfile.close();
    }

    return EXIT_SUCCESS;
}
