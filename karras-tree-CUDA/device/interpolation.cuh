#include <iterator>

namespace grace {

namespace gpu {

// Requires x in [0, N_table)
template <typename Real, typename TableIter>
GRACE_DEVICE Real lerp(Real x, TableIter table, int N_table)
{
    typedef typename std::iterator_traits<TableIter>::value_type TableReal;

    int x_idx = static_cast<int>(x);
    if (x_idx >= N_table - 1) {
        x = static_cast<TableReal>(N_table - 1);
        x_idx = N_table - 2;
    }
    Real integral = fma(table[x_idx + 1] - table[x_idx],
                        static_cast<TableReal>(x - x_idx),
                        table[x_idx]);
    return integral;
}

} // namespace gpu

} // namespace grace
