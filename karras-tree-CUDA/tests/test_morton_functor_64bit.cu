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
    /* Compare thrust::transform to a CPU loop, for 63-bit keys. */
    /*************************************************************/

    typedef grace::Vector3<double> Vector3d;


    /* Generate random doubles in [0,1) on CPU and find the keys. */

    thrust::uniform_real_distribution<double> u01_d(0,1);

    thrust::host_vector<Vector3d> h_random_d(N);
    thrust::host_vector<UInteger64> h_morton_64(N);

    for (unsigned int i=0; i<N; i++) {
        h_random_d[i].x = u01_d(rng);
        h_random_d[i].y = u01_d(rng);
        h_random_d[i].z = u01_d(rng);

        UInteger64 ix = (UInteger64) (h_random_d[i].x * ((1u << 21) -1));
        UInteger64 iy = (UInteger64) (h_random_d[i].y * ((1u << 21) -1));
        UInteger64 iz = (UInteger64) (h_random_d[i].z * ((1u << 21) -1));

        h_morton_64[i] = grace::morton_key_63bit(ix, iy, iz);
    }


    /* Calculate morton keys on GPU from the same floats. */

    // morton_key_functor requires AABB information.
    Vector3d bottom_d(0., 0., 0.);
    Vector3d top_d(1., 1., 1.);

    thrust::device_vector<Vector3d> d_random_d = h_random_d;
    thrust::device_vector<UInteger64> d_morton_64(N);
    thrust::transform(d_random_d.begin(),
                      d_random_d.begin() + N,
                      d_morton_64.begin(),
                      grace::morton_key_functor<UInteger64, double>(bottom_d, top_d) );


    /* Verify results are the same. */

    correct = true;
    thrust::host_vector<UInteger64> h_d_morton_copy_64 = d_morton_64;
    for (unsigned int i=0; i<N; i++) {
        if (h_morton_64[i] != h_d_morton_copy_64[i]) {
            std::cout << "Device morton key != host morton key (63-bit)!" << std::endl;
            std::cout << "Host floats: x: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_random_d[i].x << std::endl;
            std::cout << "             y: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_random_d[i].y << std::endl;
            std::cout << "             z: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_random_d[i].z << std::endl;
            std::cout << "Host key:   " << (std::bitset<64>) h_morton_64[i] << std::endl;
            std::cout << "Device key: " << (std::bitset<64>) h_d_morton_copy_64[i] << std::endl;
            correct = false;
        }
    }

    if (correct) {
        std::cout << "All " << N << " 63-bit GPU and host keys match!" << std::endl;
    }

    return 0;
}
