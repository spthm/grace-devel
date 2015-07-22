#include "interval.h"

#include <gmpxx.h>
#include <algorithm>

template <>
bool Interval<mpq_class>::contains(const mpq_class& c) const
{
    int ac = cmp(_a, c);
    int cb = cmp(c, _b);

    return ac <= 0 && cb <= 0;
}
