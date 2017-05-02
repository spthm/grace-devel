#pragma once

#include "interval.h"

// We need to pre-declare all friend functions of a templated class which are
// themsevles templates.
template <typename T> class VertexQuadratic; // Pre-declare class itself...
template <typename T> std::ostream& operator<<(std::ostream& ostrm, const VertexQuadratic<T>& eqn);

/* Quadratic of the form ax^2 + bx + c, stored internally as a(x - h)^2 + k.
 * This form avoids the dependency problem when evaluating intervals.
 *
 * The class is templated for use with any numeric type.
 * The evaluate() methods only accept inputs of said numeric type, or Intervals
 * of that type, in order to avoid ambiguity in the return type.
 * Implicit type conversion may therefore occur during an evaluate() call.
 *
 * Conversion from the canonical (a, b, c) to vertex quadratic (a, h, k)
 * representations is in general subject to rounding error.
 */
template <typename T>
class VertexQuadratic
{
protected:
    T _a, _h, _k;

public:
    VertexQuadratic(): _a(1), _h(0), _k(0) {}
    VertexQuadratic(const T& a, const T& b, const T& c);

    T evaluate(const T& x) const
    {
        return _a * (x - _h) * (x - _h) + _k;
    }

    // Guaranteed to be exact, rounding error notwithstanding.
    Interval<T> evaluate(const Interval<T>& x) const
    {
        // For an interval x, x*x may be wider than the true range.
        return _a * square(x - _h) + _k;
    }

    bool root_exists(const Interval<T>& interv) const
    {
        Interval<T> eqn_interv = evaluate(interv);
        return eqn_interv.contains(0);
    }

    friend std::ostream& operator<< <> (std::ostream& ostrm,
                                        const VertexQuadratic<T>& eqn);
};

template <typename T>
VertexQuadratic<T>::VertexQuadratic(const T& a, const T& b, const T& c)
{
    _a = a;
    _h = -b / (2 * a);
    _k = c - b * b / (4 * a);
}

template <typename T>
std::ostream& operator<<(std::ostream& ostrm, const VertexQuadratic<T>& eqn)
{
    ostrm << eqn._a << "(x - " << eqn._h << ")^2 + " << eqn._k;
    return ostrm;
}
