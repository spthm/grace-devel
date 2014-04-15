#include <iostream>
#include <bitset>

#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../kernels/bits.cuh"

class bit_prefix_functor
{
public:

    __device__ grace::uinteger32 operator() (const grace::uinteger32& a,
                                             const grace::uinteger32& b) {

        return grace::gpu::bit_prefix_length(a, b);
    }
};

int main(int argc, char* argv[]) {

    /**************************************/
    /* 32-bit edge-case key calculations. */
    /**************************************/
    //
    // a = 1011 1001 0001 1011 1111 0101 0110 1101 = 3105617261
    // b = 0011 1001 0001 1011 1111 0101 0110 1101 = 958133613
    // length of common prefix = 0

    // a = 1011 1001 0001 1011 1111 0101 0110 1101 = 3105617261
    // b = 1011 1001 0001 1011 1111 0101 0110 1101 = 3105617261
    // length of common prefix = 32

    grace::uinteger32 As[2] = {3105617261, 3105617261};
    grace::uinteger32 Bs[2] = {958133613, 3105617261};
    grace::uinteger32 actuals[3] = {0, 32};

    thrust::device_vector<grace::uinteger32> d_As(As, As+2);
    thrust::device_vector<grace::uinteger32> d_Bs(Bs, Bs+2);

    thrust::device_vector<grace::uinteger32> d_calculated(2);
    thrust::transform(d_As.begin(), d_As.end(),
                      d_Bs.begin(),
                      d_calculated.begin(),
                      bit_prefix_functor());
    thrust::host_vector<grace::uinteger32> h_calculated = d_calculated;

    std::cout << "Testing two edge cases (prefix length = 0, 32):\n"
              << std::endl;
    for (int i=0; i<2; i++) {
        std::cout << "a: " << (std::bitset<32>) As[i] << std::endl;
        std::cout << "b: " << (std::bitset<32>) Bs[i] << std::endl;
        std::cout << "Actual prefix length:     " << actuals[i] << std::endl;
        std::cout << "Calculated prefix length: " << h_calculated[i]
                  << std::endl;
        std::cout << std::endl;
    }


    /************************************/
    /* 32-bit ranadom key calculations. */
    /************************************/

    unsigned int  N = 10000;
    std::cout << "Testing " << N << " random 32-bit integers...\n" << std::endl;


    /* Generate N random unsigned integers. */

    thrust::default_random_engine rng(1234);
    thrust::uniform_int_distribution<grace::uinteger32> rand_uint;

    thrust::host_vector<grace::uinteger32> h_As(N);
    thrust::host_vector<grace::uinteger32> h_Bs(N);
    for (int i=0; i<N; i++) {
        h_As[i] = rand_uint(rng);
        h_Bs[i] = rand_uint(rng);
    }


    /* Calculate bit prefixes on GPU. */

    d_As = h_As;
    d_Bs = h_Bs;
    d_calculated.resize(N);
    thrust::transform(d_As.begin(), d_As.end(),
                      d_Bs.begin(),
                      d_calculated.begin(),
                      bit_prefix_functor());
    h_calculated = d_calculated;


    /* Compare device results to host bit-prefix function. */

    bool correct = true;
    for (int i=0; i<N; i++) {
        grace::uinteger32 prefix_length = grace::bit_prefix_length(h_As[i],
                                                                   h_Bs[i]);

        if (prefix_length != h_calculated[i]) {
            std::cout << "Device prefix length key != host prefix length!"
                       << std::endl;
            std::cout << "a: " << (std::bitset<32>) h_As[i] << std::endl;
            std::cout << "b: " << (std::bitset<32>) h_Bs[i] << std::endl;
            std::cout << "Host prefix length: " << prefix_length << std::endl;
            std::cout << "Device prefix length: " << h_calculated[i]
                      << std::endl;
            std::cout << std::endl;
            correct = false;
        }
    }

    if (correct) {
        std::cout << "All " << N << " GPU and host prefix lengths match!"
                  << std::endl;
    }

    return 0;
}
