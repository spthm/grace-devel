#include "grace/generic/bits.h"
#include "grace/generic/morton.h"

#include <cstdlib>
#include <iostream>

int main(void)
{
    typedef grace::uinteger32 KeyT;
    bool correct = true;

    // x = 0100110101 = 309
    // y = 1110101110 = 942
    // z = 1001101011 = 619
    // => spaced_x =   0001000000001001000001000001 = 16814145
    // => spaced_y =  1001001000001000001001001000  = 153125448
    // => spaced_z = 1000000001001000001000001001   = 134513161
    // => key      = 110011010100111001110011110101 = 861117685

    KeyT x = 309;
    KeyT y = 942;
    KeyT z = 619;
    KeyT ref_spaced_x = 16814145;
    KeyT ref_spaced_y = 153125448;
    KeyT ref_spaced_z = 134513161;
    KeyT ref_key = 861117685;

    KeyT spaced_x = grace::detail::space_by_two_10bit(x);
    KeyT spaced_y = grace::detail::space_by_two_10bit(y);
    KeyT spaced_z = grace::detail::space_by_two_10bit(z);
    KeyT key = grace::morton_key(x, y, z);

    if ((spaced_x != ref_spaced_x) ||
        (spaced_y != ref_spaced_y) ||
        (spaced_z != ref_spaced_z))
    {
        std::cout << "FAILED space_by_two_10bit" << std::endl;
        correct = false;
    }

    if (key != ref_key)
    {
        std::cout << "FAILED morton_key<grace::uinteger32>()" << std::endl;
        correct = false;
    }

    if (correct) {
        std::cout << "PASSED" << std::endl;
    }

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
