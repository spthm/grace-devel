#include "print_table.h"

#include <iostream>
#include <iomanip>
#include <sstream>

class ThreeSF
{
private:
    double value;
    int minwidth;
public:
    explicit ThreeSF(const double value, const int minwidth=0)
        : value(value), minwidth(minwidth) {}
    explicit ThreeSF(const float value, const int minwidth=0)
        : value(value), minwidth(minwidth) {}

    double get_value() const { return value; }
    int get_minwidth() const { return minwidth; }
};

std::ostream& operator<<(std::ostream& os, const ThreeSF& lhs)
{
    const double v = lhs.get_value();
    const int minwidth = lhs.get_minwidth();
    std::ostringstream v_ss;
    // Must be wide enough to always contain the 'last' digit (see below).
    v_ss << std::fixed << std::setprecision(9) << v;
    std::string v_str = v_ss.str();

    int length;
    if (v >= 1000.) {
        int v_sig, v_round;
        std::istringstream(v_str.substr(0, 3)) >> v_sig;
        std::istringstream(v_str.substr(3, 1)) >> v_round;
        if (v_round >= 5) { ++v_sig; }

        int dpidx = v_str.find('.');
        if (dpidx == std::string::npos) { dpidx = v_str.length(); }

        os << v_sig << std::string(dpidx - 3, '0');

        length = dpidx;
    }
    else
    {
        int last, n;

        // This is no good for values smaller than 0.01.
        if (v < 0.1)        { last = 6; n = 1; }
        else if (v < 1.)    { last = 5; n = 1; }
        else if (v < 10.)   { last = 4; n = 1; }
        else if (v < 100.)  { last = 4; n = 1; }
        else /* v < 1000 */ { last = 3; n = 2; }

        // The obvious os << std::setprecision(p) << v may round the last digit.
        os << v_str.substr(0, last) << "(" << v_str.substr(last, n) << ")";
        length = last + n + 2;
    }

    // Pad end so total length is at least minwidth.
    std::string pad = length < minwidth ? std::string(minwidth - length, ' ') : "";
    return os << pad;
}

void cout_init()
{
    std::cout.fill(' ');
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
}

void print_header(const size_t N, const unsigned int bitmask)
{
    std::cout << "                         N = " << std::setw(10) << N
              << std::endl;
    if (bitmask == 1u) {
    std::cout << "  RNG  |   P(x = ... 1)  |          State            | Tgen (ms)";
    }
    else if (bitmask == 3u) {
    std::cout << "  RNG  |  P(x = ... 11)  |          State            | Tgen (ms)";
    }
    else if (bitmask == 7u) {
    std::cout << "  RNG  |  P(x = ... 111) |          State            | Tgen (ms)";
    }
    else if (bitmask == 15u) {
    std::cout << "  RNG  | P(x = ... 1111) |          State            | Tgen (ms)";
    }
    else  {
    std::cout << "  RNG  |   P(x & M = M)  |          State            | Tgen (ms)";
    }
    std::cout << std::endl
              << "       |                 | Size (MiB) |  Tinit (ms)  |"
              << std::endl
              << "-------|-----------------|------------|--------------|----------"
              << std::endl;
}

void print_row(const int rng, const double p, const size_t size_bytes,
               const double tinit, const double tgen)
{
    const double size = size_bytes / 1024. / 1024.;
    if (rng == PHILOX)
        std::cout << std::setw(7) << "PHILOX ";
    else if (rng == XORWOW)
        std::cout << std::setw(7) << "XORWOW ";
    else if (rng == MRG32)
        std::cout << std::setw(7) << "MRG32 ";
    else
        // Throw.
        return;

    std::cout << "|    " << ThreeSF(p, 9) << "    ";
    std::cout << "|  " << ThreeSF(size, 8) << "  ";
    std::cout << "|  " << ThreeSF(tinit, 9) << "   ";
    std::cout << "|  " << ThreeSF(tgen, 8);
    std::cout << std::endl;
}

void print_footer()
{
    std::cout << std::endl << std::endl;
}
