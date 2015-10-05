#include <iostream>
#include <bitset>

#include "../device/bits.cuh"
#include "../device/morton.cuh"

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

    typedef grace::uinteger32 Key;


    /* 10-bit x, y, z integers and binary representation. */

    Key x = 309;
    Key y = 942;
    Key z = 619;


    /* 10-bit spaced-by-two integers and binary representation. */

    Key spaced_x = 16814145;
    Key spaced_y = 153125448;
    Key spaced_z = 134513161;
    Key key = 861117685;

    Key my_spaced_x = grace::bits::space_by_two_10bit(x);
    Key my_spaced_y = grace::bits::space_by_two_10bit(y);
    Key my_spaced_z = grace::bits::space_by_two_10bit(z);


    /* 30-bit keys and binary representation. */

    Key my_key = my_spaced_x | my_spaced_y << 1 |
                               my_spaced_z << 2;
    Key my_key_2 = grace::morton::morton_key(x, y, z);

    /* Print everything. */

    std::cout << "30-bit Morton key results:" << std::endl << std::endl;
    std::cout << "x:                     " << (std::bitset<32>) x << std::endl;
    std::cout << "Spaced x:              " << (std::bitset<32>) spaced_x
              << std::endl;
    std::cout << "space_by_two_10bit(x): " << (std::bitset<32>) my_spaced_x
              << std::endl << std::endl;
    std::cout << "y:                     " << (std::bitset<32>) y
              << std::endl;
    std::cout << "Spaced y:              " << (std::bitset<32>) spaced_y
              << std::endl;
    std::cout << "space_by_two_10bit(y): " << (std::bitset<32>) my_spaced_y
              << std::endl << std::endl;
    std::cout << "z:                     " << (std::bitset<32>) z << std::endl;
    std::cout << "Spaced z:              " << (std::bitset<32>) spaced_z
              << std::endl;
    std::cout << "space_by_two_10bit(z): " << (std::bitset<32>) my_spaced_z
              << std::endl << std::endl;
    std::cout << "Key:                   " << (std::bitset<32>) key
              << std::endl;
    std::cout << "space_by_two_10bit:    " << (std::bitset<32>) my_key
              << std::endl;
    std::cout << "morton_key_30bit:      " << (std::bitset<32>) my_key_2
              << std::endl << std::endl;

    bool correct = true;
    if (my_spaced_x != spaced_x) {
        std::cout << "x failed!" << std::endl;
        correct = false;
    }
    if (my_spaced_y != spaced_y) {
        std::cout << "y failed!" << std::endl;
        correct = false;
    }
    if (my_spaced_z != spaced_z) {
        std::cout << "z failed!" << std::endl;
        correct = false;
    }
    if (correct) {
        std::cout << std::endl << "All correct!" << std::endl;
    }

    return 0;
}
