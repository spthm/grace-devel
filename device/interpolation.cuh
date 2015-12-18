#include <iterator>

namespace grace {

namespace interp {

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

    TableReal y0 = table[x_idx];
    TableReal y1 = table[x_idx + 1];
    TableReal t = static_cast<TableReal>(x) - x_idx;

    // y = t * (y1 - y0) + y0 = t * y1 - t * y0 + y0
    // In the general case, the has error <~ 1 ulp
    // Real y = fma(t, y1, fma(-t, y0, y0));

    // Sterbenz lemma: floating-point a - b is exact for 0.5*a <= b <= 2*a
    // This typically holds for adjacent values in a lookup table, so we choose
    // the below form over the above. It has error 0.5 ulp (error of FMA).
    Real y = fma(t, y1 - y0, y0);
    return y;
}

} // namespace interp

} // namespace grace
