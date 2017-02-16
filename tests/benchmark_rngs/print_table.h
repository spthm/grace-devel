#include <stddef.h>

enum
{
    PHILOX,
    XORWOW,
    MRG32,
    num_generators
};

void cout_init();
void print_header(const size_t N);
void print_row(const int rng, const double p, const size_t size_bytes,
               const double tinit, const double tgen);
void print_footer();
