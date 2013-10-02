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
    /* Compare thrust::transform to a CPU loop, for 30-bit keys. */
    /*************************************************************/

    typedef grace::Vector3<float> Vector3f;

    /* Generate random floats in [0,1) on CPU and find the keys. */

    thrust::default_random_engine rng(1234);
    thrust::uniform_real_distribution<float> u01(0,1);

    unsigned int N = 10000;
    thrust::host_vector<Vector3f> h_random(N);
    thrust::host_vector<UInteger32> h_morton(N);

    for (unsigned int i=0; i<N; i++) {
        h_random[i].x = u01(rng);
        h_random[i].y = u01(rng);
        h_random[i].z = u01(rng);

        UInteger32 ix = (UInteger32) (h_random[i].x * 1023);
        UInteger32 iy = (UInteger32) (h_random[i].y * 1023);
        UInteger32 iz = (UInteger32) (h_random[i].z * 1023);

        h_morton[i] = grace::morton_key_30bit(ix, iy, iz);
    }

    /* Calculate morton keys on GPU from the same floats. */

    // morton_key_functor requires AABB information.
    Vector3f bottom(0., 0., 0.);
    Vector3f top(1., 1., 1.);

    thrust::device_vector<Vector3f> d_random = h_random;
    thrust::device_vector<UInteger32> d_morton(N);
    thrust::transform(d_random.begin(),
                      d_random.begin() + N,
                      d_morton.begin(),
                      grace::morton_key_functor<UInteger32, float>(bottom, top) );

    /* Verify results are the same. */

    bool correct = true;
    thrust::host_vector<UInteger32> h_d_morton_copy = d_morton;
    for (unsigned int i=0; i<N; i++) {
        if (h_morton[i] != h_d_morton_copy[i]) {
            std::cout << "Device morton key != host morton key!" << std::endl;
            std::cout << "Host floats: x: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_random[i].x << std::endl;
            std::cout << "             y: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_random[i].y << std::endl;
            std::cout << "             z: " << std::fixed << std::setw(6)
                      << std::setprecision(6) << std::setfill('0')
                      << h_random[i].z << std::endl;
            std::cout << "Host key:   " << (std::bitset<32>) h_morton[i] << std::endl;
            std::cout << "Device key: " << (std::bitset<32>) h_d_morton_copy[i] << std::endl;
            correct = false;
        }
    }

    if (correct) {
        std::cout << "All GPU keys match all host keys!" << std::endl;
    }

    return 0;
}
