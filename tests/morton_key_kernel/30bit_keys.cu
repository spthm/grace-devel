#include "grace/cuda/detail/kernels/morton.cuh"
#include "grace/generic/morton.h"
#include "grace/generic/functors/centroid.h"

#include "helper/random.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cstdlib>
#include <iostream>
#include <bitset>
#include <iomanip>

int main(int argc, char* argv[])
{
    typedef grace::uinteger32 KeyT;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(6);
    std::cout.fill('0');

    /*************************************************************/
    /* Compare morton_key_kernel to a CPU loop, for 63-bit keys. */
    /*************************************************************/

    size_t N = 10000;
    bool verbose = false;
    if (argc > 1) {
        N = (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        verbose = (std::string(argv[2]) == "true") ? true : false;
    }

    // Generate N random points with single precision co-ordinates in [-1, 1).
    float3 top = make_float3(1., 1., 1.);
    float3 bot = make_float3(-1., -1., -1.);
    thrust::host_vector<float3> h_points(N);
    thrust::transform(thrust::counting_iterator<size_t>(0),
                      thrust::counting_iterator<size_t>(N),
                      h_points.begin(),
                      random_real3_functor<float3>());
    thrust::device_vector<float3> d_points = h_points;

    // Compute keys on host.
    thrust::host_vector<KeyT> h_keys(N);
    const KeyT MAX_KEY = 1023u;
    for (size_t i = 0; i < N; ++i) {
        KeyT ux = static_cast<KeyT>((h_points[i].x - bot.x) * MAX_KEY);
        KeyT uy = static_cast<KeyT>((h_points[i].y - bot.y) * MAX_KEY);
        KeyT uz = static_cast<KeyT>((h_points[i].z - bot.z) * MAX_KEY);

        h_keys[i] = grace::morton_key(ux, uy, uz);
    }

    // Compute keys on device.
    thrust::device_vector<KeyT> d_keys(N);
    grace::morton_keys(d_points, bot, top, d_keys, grace::CentroidSphere());

    // Check device keys against host keys.
    int errors = 0;
    thrust::host_vector<KeyT> h_d_keys = d_keys;
    for (size_t i = 0; i < N; ++i)
    {
        KeyT h_key = h_keys[i];
        KeyT d_key = h_d_keys[i];
        if (h_key != d_key)
        {
            ++errors;

            if (!verbose) {
                continue;
            }

            std::cout << "host morton key != device morton key" << std::endl
                      << "(x, y, z): " << " ("
                      << std::setw(8) << h_points[i].x << ", "
                      << std::setw(8) << h_points[i].y << ", "
                      << std::setw(8) << h_points[i].z << ")" << std::endl
                      << "Host key:   " << std::bitset<64>(h_key)
                      << std::endl
                      << "Device key: " << std::bitset<64>(d_key)
                      << std::endl
                      << "Diff bits:  " << std::bitset<64>(h_key ^ d_key)
                      << std::endl << std::endl;
        }
    }

    if (errors != 0 && verbose) {
        std::cout << std::endl;
    }

    if (errors == 0) {
        std::cout << "PASSED" << std::endl;
    }
    else {
        std::cout << errors << " of " << N << " keys did not match host"
                  << std::endl
                  << "FAILED" << std::endl;
    }

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
