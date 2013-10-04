#include <iostream>
#include <bitset>

#include "../types.h"
#include "../kernels/bits.cuh"
#include "../kernels/morton.cuh"

int main(int argc, char* argv[]) {

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
    std::cout << "morton_key_63bit:         " << (std::bitset<64>) my_key_2_64 << "\n" << std::endl;

    return 0;
}
