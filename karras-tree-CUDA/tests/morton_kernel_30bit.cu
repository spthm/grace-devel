#include <iostream>
#include <bitset>
#include <iomanip>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../types.h"
#include "../kernels/morton.cuh"

int main(int argc, char* argv[]) {

    /*************************************************************/
    /* Compare morton_key_kernel to a CPU loop, for 30-bit keys. */
    /*************************************************************/

    typedef grace::Vector3<float> Vector3f;


    /* Generate random floats in [0,1) on CPU and find the keys. */

    unsigned int N;
    if (argc > 1)
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
    else
        N = 10000;

    thrust::default_random_engine rng(1234);
    thrust::uniform_real_distribution<float> u01(0,1);

    thrust::host_vector<float> h_x_random(N);
    thrust::host_vector<float> h_y_random(N);
    thrust::host_vector<float> h_z_random(N);
    thrust::host_vector<grace::uinteger32> h_morton(N);

    for (unsigned int i=0; i<N; i++) {
        h_x_random[i] = u01(rng);
        h_y_random[i] = u01(rng);
        h_z_random[i] = u01(rng);

        grace::uinteger32 ix = (grace::uinteger32) (h_x_random[i] * 1023);
        grace::uinteger32 iy = (grace::uinteger32) (h_y_random[i] * 1023);
        grace::uinteger32 iz = (grace::uinteger32) (h_z_random[i] * 1023);

        h_morton[i] = grace::morton_key(ix, iy, iz);
    }


    /* Calculate morton keys on GPU from the same floats. */

    // Generating morton keys requires AABB information.
    Vector3f bottom(0., 0., 0.);
    Vector3f top(1., 1., 1.);

    thrust::device_vector<float> d_x_random = h_x_random;
    thrust::device_vector<float> d_y_random = h_y_random;
    thrust::device_vector<float> d_z_random = h_z_random;
    thrust::device_vector<grace::uinteger32> d_morton(N);
    grace::morton_keys(d_x_random, d_y_random, d_z_random,
                       d_morton, bottom, top);


    /* Verify results are the same. */

    unsigned int err_count = 0;
    thrust::host_vector<grace::uinteger32> h_d_morton_copy = d_morton;
    for (unsigned int i=0; i<N; i++) {
        if (h_morton[i] != h_d_morton_copy[i]) {
            std::cout << "Device morton key != host morton key!" << std::endl;
            std::cout << "Host floats: x: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_x_random[i] << std::endl;
            std::cout << "             y: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_y_random[i] << std::endl;
            std::cout << "             z: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_z_random[i] << std::endl;
            std::cout << "Host key:   " << (std::bitset<32>) h_morton[i] << std::endl;
            std::cout << "Device key: " << (std::bitset<32>) h_d_morton_copy[i] << std::endl;
            err_count++;
        }
    }

    if (err_count == 0) {
        std::cout << "All " << N << " GPU and host keys match!" << std::endl;
    }
    else{
        std::cout << err_count << " keys were incorrect, out of " << N
                  << std::endl;
    }

    return 0;
}
