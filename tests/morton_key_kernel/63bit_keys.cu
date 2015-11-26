#include <iostream>
#include <bitset>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

#include "device/morton.cuh"
#include "kernels/build_sph.cuh"

int main(int argc, char* argv[]) {

    /*************************************************************/
    /* Compare morton_key_kernel to a CPU loop, for 63-bit keys. */
    /*************************************************************/

    typedef grace::uinteger64 Key;

    /* Generate random doubles in [0,1) on CPU and find the keys. */

    unsigned int N = 10000;
    if (argc > 1)
        N = (unsigned int) std::strtol(argv[1], NULL, 10);

    thrust::host_vector<double4> h_points(N);
    thrust::host_vector<Key> h_keys(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_points.begin(),
                      grace::random_double4_functor());

    for (int i=0; i<N; i++) {
        Key ux = (Key) (h_points[i].x * ((1u << 21) - 1));
        Key uy = (Key) (h_points[i].y * ((1u << 21) - 1));
        Key uz = (Key) (h_points[i].z * ((1u << 21) - 1));

        h_keys[i] = grace::morton::morton_key(ux, uy, uz);
    }


    /* Calculate morton keys on GPU from the same doubles. */

    // Generating morton keys requires AABB information.
    float3 top = make_float3(1., 1., 1.);
    float3 bot = make_float3(0., 0., 0.);

    thrust::device_vector<double4> d_points = h_points;
    thrust::device_vector<Key> d_keys(N);

    grace::morton_keys_sph(d_points, top, bot, d_keys);


    /* Verify results are the same. */

    unsigned int err_count = 0;
    thrust::host_vector<Key> h_d_keys = d_keys;
    for (unsigned int i=0; i<N; i++) {
        if (h_keys[i] != h_d_keys[i]) {
            std::cout << "Device morton key != host morton key!" << std::endl;
            std::cout << "Host doubles: x: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_points[i].x << std::endl;
            std::cout << "             y: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_points[i].y << std::endl;
            std::cout << "             z: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_points[i].z << std::endl;
            std::cout << "Host key:   " << (std::bitset<64>) h_keys[i]
                      << std::endl;
            std::cout << "Device key: " << (std::bitset<64>) h_d_keys[i]
                      << std::endl;
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
