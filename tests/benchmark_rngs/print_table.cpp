#include "print_table.h"

#include <iostream>
#include <iomanip>
#include <sstream>

class ThreeSF
{
private:
    double value;
public:
    explicit ThreeSF(const double value) : value(value) {}
    explicit ThreeSF(const float value) : value(value) {}

    double get_value() const { return value; }
};

std::ostream& operator<<(std::ostream& os, const ThreeSF& lhs)
{
    const double v = lhs.get_value();
    std::ostringstream v_ss;
    // Must be wide enough to always contain the 'last' digit (see below).
    v_ss << std::fixed << std::setprecision(9) << v;
    std::string v_str = v_ss.str();
    int last, n;

    if (v >= 1000.) {
        int v_sig, v_round;
        std::istringstream(v_str.substr(0, 3)) >> v_sig;
        std::istringstream(v_str.substr(3, 1)) >> v_round;
        if (v_round >= 5) { ++v_sig; }

        int n = v_str.find('.');
        if (n == std::string::npos) { n = v_str.length(); }

	// If the string is shorter than 7 characters, i.e. shorter than
        // x.xx(x), xx.x(x) and xxx(.x), pad with whitespace up to that length.
	std::string pad = n < 7 ? std::string(7 - n, ' ') : "";
        return os << v_sig << std::string(n - 3, '0') << pad;
    }
    else if (v < 0.1) { last = 6; n = 1; }
    else if (v < 1.) { last = 5; n = 1; }
    else if (v < 10.) { last = 4; n = 1; }
    else if (v < 100.) { last = 4; n = 1; }
    else if (v < 1000.) { last = 3; n = 2; }
    else { return os << v; }

    // The obvious os << std::setprecision(p) << v may round the last digit.
    return os << v_str.substr(0, last) << "(" << v_str.substr(last, n) << ")";
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
    std::cout << "  RNG  |   P(x = ... 1)  |          State          | Tgen (ms)";
    }
    else if (bitmask == 3u) {
    std::cout << "  RNG  |  P(x = ... 11)  |          State          | Tgen (ms)";
    }
    else if (bitmask == 7u) {
    std::cout << "  RNG  |  P(x = ... 111) |          State          | Tgen (ms)";
    }
    else if (bitmask == 15u) {
    std::cout << "  RNG  | P(x = ... 1111) |          State          | Tgen (ms)";
    }
    else  {
    std::cout << "  RNG  |   P(x & M = M)  |          State          | Tgen (ms)";
    }
    std::cout << std::endl
              << "       |                 | Size (MiB) | Tinit (ms) |"
              << std::endl
              << "-------|-----------------|------------|------------|----------"
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

    // Width of 6 good down to p = 0.0xxx.
    std::cout << "|    " << std::setw(6) << ThreeSF(p) << "    ";
    std::cout << "|  " << std::setw(5) << ThreeSF(size) << "  ";
    std::cout << "|  " << ThreeSF(tinit) << "   ";
    std::cout << "|  " << ThreeSF(tgen);
    std::cout << std::endl;
}

void print_footer()
{
    std::cout << std::endl << std::endl;
}
