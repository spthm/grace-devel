#include "grace/cuda/device/bits.cuh"
#include "grace/cuda/device/morton.cuh"

#include <cstdlib>
#include <iostream>

int main(void)
{
    typedef grace::uinteger64 KeyT;
    bool correct = true;

    // x = 101001101010100110101 = 1365301
    // y = 111101011101110101110 = 2014126
    // z = 110011010111001101011 = 1683051
    // => spaced_x =   1000001000000001001000001000001000001000000001001000001000001 = 1170975555344961601
    // => spaced_y =  1001001001000001000001001001000001001001000001000001001001000  = 1317338702596309576
    // => spaced_z = 1001000000001001000001000001001001000000001001000001000001001   = 1297353911585505801
    // => key      = 111110011010100111001110011110101110011010100111001110011110101 = 8995068606879603957

    KeyT x = 1365301;
    KeyT y = 2014126;
    KeyT z = 1683051;
    KeyT ref_spaced_x = 1170975555344961601;
    KeyT ref_spaced_y = 1317338702596309576;
    KeyT ref_spaced_z = 1297353911585505801;
    KeyT ref_key = 8995068606879603957;

    KeyT spaced_x = grace::bits::space_by_two_21bit(x);
    KeyT spaced_y = grace::bits::space_by_two_21bit(y);
    KeyT spaced_z = grace::bits::space_by_two_21bit(z);
    KeyT key = grace::morton::morton_key(x, y, z);

    if ((spaced_x != ref_spaced_x) ||
        (spaced_y != ref_spaced_y) ||
        (spaced_z != ref_spaced_z))
    {
        std::cout << "FAILED space_by_two_21bit" << std::endl;
        correct = false;
    }
    else if (key != ref_key)
    {
        std::cout << "FAILED morton_key<grace::uinteger64>()" << std::endl;
        correct = false;
    }

    if (correct) {
        std::cout << "PASSED" << std::endl;
    }

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
