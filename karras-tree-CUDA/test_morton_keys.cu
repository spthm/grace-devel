#include <iostream>
#include <bitset>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "types.h"
#include "kernels/bits.cuh"
#include "kernels/morton.cuh"

int main(int argc, char* argv[]) {

    /****************************/
    /* 30-bit key calculations. */
    /****************************/
    //
    // x = 0100110101 = 309
    // y = 1110101110 = 942
    // z = 1001101011 = 619
    // => spaced_x =   0001000000001001000001000001 = 16814145
    // => spaced_y =  1001001000001000001001001000  = 153125448
    // => spaced_z = 1000000001001000001000001001   = 134513161
    // => key      = 110011010100111001110011110101

    /* 10-bit x, y, z integers and binary representation. */

    UInteger32 x = 309;
    UInteger32 y = 942;
    UInteger32 z = 619;


    /* 10-bit spaced-by-two integers and binary representation. */

    UInteger32 spaced_x = 16814145;
    UInteger32 spaced_y = 153125448;
    UInteger32 spaced_z = 134513161;
    UInteger32 key = 861117685;

    UInteger32 my_spaced_x = grace::space_by_two_10bit(x);
    UInteger32 my_spaced_y = grace::space_by_two_10bit(y);
    UInteger32 my_spaced_z = grace::space_by_two_10bit(z);


    /* 30-bit keys and binary representation. */

    UInteger32 my_key = my_spaced_x | my_spaced_y << 1 | my_spaced_z << 2;
    UInteger32 my_key_2 = grace::morton_key_30bit(x, y, z);



    /****************************/
    /* 63-bit key calculations. */
    /****************************/
    //
    // x_64 = 101001101010100110101 = 1365301
    // y_64 = 111101011101110101110 = 2014126
    // z_64 = 110011010111001101011 = 1683051
    // => spaced_x_64 =   1000001000000001001000001000001000001000000001001000001000001 = 1170975555344961601
    // => spaced_y_64 =  1001001001000001000001001001000001001001000001000001001001000  = 1317338702596309576
    // => spaced_z_64 = 1001000000001001000001000001001001000000001001000001000001001   = 1297353911585505801
    // => key_64      = 111110011010100111001110011110101110011010100111001110011110101 = 8995068606879603957

    /* 10-bit x, y, z integers and binary representation. */

    UInteger64 x_64 = 1365301;
    UInteger64 y_64 = 2014126;
    UInteger64 z_64 = 1683051;


    /* 10-bit spaced-by-two integers and binary representation. */

    UInteger64 spaced_x_64 = 1170975555344961601;
    UInteger64 spaced_y_64 = 1317338702596309576;
    UInteger64 spaced_z_64 = 1297353911585505801;
    UInteger64 key_64 = 8995068606879603957;

    UInteger64 my_spaced_x_64 = grace::space_by_two_21bit(x_64);
    UInteger64 my_spaced_y_64 = grace::space_by_two_21bit(y_64);
    UInteger64 my_spaced_z_64 = grace::space_by_two_21bit(z_64);


    /* 30-bit keys and binary representation. */

    UInteger64 my_key_64 = my_spaced_x_64 | my_spaced_y_64 << 1 | my_spaced_z_64 << 2;
    UInteger64 my_key_2_64 = grace::morton_key_63bit(x_64, y_64, z_64);


    /* Print everything. */

    std::cout << "30-bit Morton key results:\n" << std::endl;
    std::cout << "x:                     " << (std::bitset<32>) x << std::endl;
    std::cout << "Spaced x:              " << (std::bitset<32>) spaced_x << std::endl;
    std::cout << "space_by_two_10bit(x): " << (std::bitset<32>) my_spaced_x << "\n" << std::endl;
    std::cout << "y:                     " << (std::bitset<32>) y << std::endl;
    std::cout << "Spaced y:              " << (std::bitset<32>) spaced_y << std::endl;
    std::cout << "space_by_two_10bit(y): " << (std::bitset<32>) my_spaced_y << "\n" << std::endl;
    std::cout << "z:                     " << (std::bitset<32>) z << std::endl;
    std::cout << "Spaced z:              " << (std::bitset<32>) spaced_z << std::endl;
    std::cout << "space_by_two_10bit(z): " << (std::bitset<32>) my_spaced_z << "\n" << std::endl;
    std::cout << "Key:                   " << (std::bitset<32>) key << std::endl;
    std::cout << "space_by_two_10bit:    " << (std::bitset<32>) my_key << std::endl;
    std::cout << "morton_key_30bit:      " << (std::bitset<32>) my_key_2 << "\n\n" << std::endl;

    std::cout << "63-bit Morton key results:\n" << std::endl;
    std::cout << "x_64:                     " << (std::bitset<64>) x_64 << std::endl;
    std::cout << "Spaced x_64:              " << (std::bitset<64>) spaced_x_64 << std::endl;
    std::cout << "space_by_two_21bit(x_64): " << (std::bitset<64>) my_spaced_x_64 << "\n" << std::endl;
    std::cout << "y_64:                     " << (std::bitset<64>) y_64 << std::endl;
    std::cout << "Spaced y_64:              " << (std::bitset<64>) spaced_y_64 << std::endl;
    std::cout << "space_by_two_21bit(y_64): " << (std::bitset<64>) my_spaced_y_64 << "\n" << std::endl;
    std::cout << "z_64:                     " << (std::bitset<64>) z_64 << std::endl;
    std::cout << "Spaced z_64:              " << (std::bitset<64>) spaced_z_64 << std::endl;
    std::cout << "space_by_two_21bit(z_64): " << (std::bitset<64>) my_spaced_z_64 << "\n" << std::endl;
    std::cout << "Key_64:                   " << (std::bitset<64>) key_64 << std::endl;
    std::cout << "space_by_two_21bit:       " << (std::bitset<64>) my_key_64 << std::endl;
    std::cout << "morton_key_63bit:         " << (std::bitset<64>) my_key_2_64 << std::endl;


    /* Now compare thrust::transform to a CPU loop. */

    thrust::default_random_engine rng(1234);
    thrust::uniform_real_distribution<float> u01(0,1);

    N = 10000;
    thrust::host_vector<Vector3<float>> h_random(N);
    thrust::host_vector<UInteger32> h_morton(N);

    for (unsigned int i=0; i<N; i++) {
        h_random[i].x = u01(rng);
        h_random[i].y = u01(rng);
        h_random[i].z = u01(rng);

        h_morton[i] = grace::morton_key_30bit(h_random[i].x,
                                              h_random[i].y,
                                              h_random[i].z)
    }

    thrust::device_vector<Vector3<float>> d_random = h_random;
    thrust::device_vector<UInteger32> d_morton(N);
    thrust::transform(d_random.begin(),
                      d_random.begin() + N,
                      d_morton.begin(),
                      morton_key_functor<UInteger32>() );

}
