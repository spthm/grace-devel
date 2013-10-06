#include <iostream>
#include <bitset>

#include "../types.h"
#include "../kernels/bintree_build_kernels.cuh"

__global__ void common_prefix_kernel(Integer32* is,
                                     Integer32* js,
                                     UInteger64* keys,
                                     unsigned int n_keys,
                                     int* prefix_lengths)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    prefix_lengths[tid] = grace::gpu::common_prefix(is[tid], js[tid], keys, n_keys);
}

int main(int argc, char* argv[]) {



    thrust::host_vector<Integer32> h_is(5);
    thrust::host_vector<Integer32> h_js(5);
    thrust::host_vector<UInteger64> h_keys(6);

    // i = 0
    // j = 1
    // key_i = 1011 1001 0001 1011 1111 0101 0110 1101 1011 1001 0001 1011 1111 0101 0110 1101 = 13338524572993713517
    // key_j = 1011 1001 0001 1011 1111 0101 0110 1101 1011 1001 0011 1011 1111 0101 0110 1101 = 13338524572995810669
    // length of common prefix = 42

    h_is[0] = 0;
    h_js[0] = 1;
    h_keys[0] = 13338524572993713517;
    h_keys[1] = 13338524572995810669;

    // i = 2
    // j = 3
    // key_i = 1011 1001 0001 1011 1111 0101 0110 1101 1011 1001 0001 1011 1111 0101 0110 1101 = 13338524572993713517
    // key_i = 1011 1001 0001 1011 1111 0101 0110 1101 1011 1001 0001 1011 1111 0101 0110 1101 = 13338524572993713517
    // length of common prefix = 64 + bit_prefix(2, 3) = 64 + 31 = 95

    h_is[1] = 2;
    h_js[1] = 3;
    h_keys[2] = 13338524572993713517;
    h_keys[3] = 13338524572993713517;

    // i = 2
    // j = 4
    // key_i = 1011 1001 0001 1011 1111 0101 0110 1101 1011 1001 0001 1011 1111 0101 0110 1101 = 13338524572993713517
    // key_i = 1011 1001 0001 1011 1111 0101 0110 1101 1011 1001 0001 1011 1111 0101 0110 1101 = 13338524572993713517
    // length of common prefix = 32 + bit_prefix(2, 4) = 64 + 29 = 93

    h_is[2] = 2;
    h_js[2] = 4;
    h_keys[4] = 13338524572993713517;

    // i = 2
    // j = 5
    // key_i = 1011 1001 0001 1011 1111 0101 0110 1101 1011 1001 0001 1011 1111 0101 0110 1101 = 13338524572993713517
    // key_i = 1011 1001 0001 1011 1111 0101 0110 1101 1011 1001 0001 1011 1111 0101 0110 1101 = 13338524572993713517
    // length of common prefix = 32 + bit_prefix(2, 5) = 64 + 29 = 93

    h_is[3] = 2;
    h_js[3] = 5;
    h_keys[5] = 13338524572993713517;

    // i = -1
    // j = 1
    // length of common prefix == -1

    h_is[4] = -1;
    h_js[4] = 1;

    // i = 1
    // j = 6 = n_keys
    // length of common prefix == -1

    h_is[5] = 1;
    h_js[5] = 6;

    int actuals[6] = {42, 95, 93, 93, -1, -1};

    thrust::device_vector<Integer32> d_is = h_is;
    thrust::device_vector<Integer32> d_js = h_js;
    thrust::device_vector<UInteger64> d_keys = h_keys;
    thrust::device_vector<int> d_prefix_lengths(6);

    common_prefix_kernel<<<1,d_prefix_lengths.size()>>>(
        (Integer32*)thrust::raw_pointer_cast(d_is.data()),
        (Integer32*)thrust::raw_pointer_cast(d_js.data()),
        (UInteger64*)thrust::raw_pointer_cast(d_keys.data()),
        d_keys.size(),
        (int*)thrust::raw_pointer_cast(d_prefix_lengths.data())
    );

    thrust::host_vector<int> h_prefix_lengths = d_prefix_lengths;

    std::cout << "Valid key index range results:\n" << std::endl;
    for (int i=0; i<4; i++) {
        std::cout << "i: " << h_is[i] << std::endl;
        std::cout << "j: " << h_js[i] << std::endl;
        std::cout << "keys[i]: " << (std::bitset<64>) h_keys[h_is[i]] << std::endl;
        std::cout << "keys[j]: " << (std::bitset<64>) h_keys[h_js[i]] << std::endl;
        std::cout << "Actual prefix length: " << actuals[i] << std::endl;
        std::cout << "Calculated prefix length: " << h_prefix_lengths[i] << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Invalid key index range results:\n" << std::endl;
    for (int i=4; i<6; i++) {
        std::cout << "i: " << h_is[i] << std::endl;
        std::cout << "j: " << h_js[i] << std::endl;
        std::cout << "Actual prefix length: " << actuals[i] << std::endl;
        std::cout << "Calculated prefix length: " << h_prefix_lengths[i] << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
