#pragma once

#include <gmpxx.h>

#include <iostream>

// We need to pre-declare all friend functions of a templated class which are
// themsevles templates.
template <typename T> class Interval; // Pre-declare class itself...
template <typename T> Interval<T> operator+(const Interval<T>& lhs, const Interval<T>& rhs);
template <typename T> Interval<T> operator-(const Interval<T>& lhs, const Interval<T>& rhs);
template <typename T> Interval<T> operator*(const Interval<T>& lhs, const Interval<T>& rhs);
template <typename T> Interval<T> operator/(const Interval<T>& lhs, const Interval<T>& rhs);
template <typename T> Interval<T> square(const Interval<T>& x);

/* A class for interval arithmetic. It is templated and suitable for any numeric
 * type with defined arithmetic operators.
 *
 * Interval's arithmetic operators accept non-Interval types on both the left-
 * and right-hand sides. These functions always return an Interval<T>, where T
 * is the type of the original interval.
 *
 * Note the existence of the square(Interval<T>) function.
 */
template <typename T>
class Interval
{
private:
    T _a, _b;

public:
    Interval(): _a(0), _b(0) {}
    Interval(const T a): _a(a), _b(a) { }
    Interval(const T a, const T b): _a(a), _b(b) { }

    T get_low(void) const { return _a; }
    T get_high(void) const { return _b; }
    void set_low(const T& a) { _a = a; }
    void set_high(const T& b) { _b = b; }
    T size(void) const { return _b - _a; }

    bool contains(const T&) const;
    T midpoint(void) const { return (_a + _b) / 2.0l; }

    friend Interval<T> operator+ <> (const Interval<T>& lhs,
                                     const Interval<T>& rhs);
    friend Interval<T> operator- <> (const Interval<T>& lhs,
                                     const Interval<T>& rhs);
    friend Interval<T> operator* <> (const Interval<T>& lhs,
                                     const Interval<T>& rhs);
    friend Interval<T> operator/ <> (const Interval<T>& lhs,
                                     const Interval<T>& rhs);
    friend Interval<T> square <> (const Interval<T>& x);
};

template <typename T>
bool Interval<T>::contains(const T& c) const
{
    return (_a <= c && c <= _b);
}

// Specialized, since comparison operators <, > etc. are not defined for
// mpq_class. The implementation is in inverval.cpp to avoid multiple definition
// errors!
template <>
bool Interval<mpq_class>::contains(const mpq_class& c) const;

// =============================================================================
// friends of Interval
// =============================================================================

template <typename T>
Interval<T> square(const Interval<T>& x)
{
    // x * x fails if x._a * x._b < 0, since x * x >= 0 always.
    T aa = x._a * x._a;
    T bb = x._b * x._b;
    T ab = x._a * x._b;
    return Interval<T>(std::max(std::min(std::min(aa, bb), ab), T(0)),
                       std::max(aa, bb));
}

template <typename T>
std::ostream& operator<<(std::ostream& ostrm, const Interval<T>& interval)
{
    ostrm << "[" << interval.get_low() << ", " << interval.get_high() << "]";
    return ostrm;
}

// =============================================================================
// Arithmetic operators for:
//
//     operator(Interval, Interval)
// =============================================================================

template <typename T>
Interval<T> operator+(const Interval<T>& left, const Interval<T>& right)
{
    T a = left._a + right._a;
    T b = left._b + right._b;
    return Interval<T>(a, b);
}

template <typename T>
Interval<T> operator-(const Interval<T>& left, const Interval<T>& right)
{
    T a = left._a - right._b;
    T b = left._b - right._a;
    return Interval<T>(a, b);
}

template <typename T>
Interval<T> operator*(const Interval<T>& left, const Interval<T>& right)
{
    T aa = left._a * right._a;
    T ab = left._a * right._b;
    T ba = left._b * right._a;
    T bb = left._b * right._b;
    return Interval<T>(std::min(std::min(aa, ab), std::min(ba, bb)),
                       std::max(std::max(aa, ab), std::max(ba, bb)));
}

template <typename T>
Interval<T> operator/(const Interval<T>& left, const Interval<T>& right)
{
    // If right contains 0, the result depends on the type.
    T aa = left._a / right._a;
    T ab = left._a / right._b;
    T ba = left._b / right._a;
    T bb = left._b / right._b;
    return Interval<T>(std::min(std::min(aa, ab), std::min(ba, bb)),
                       std::max(std::max(aa, ab), std::max(ba, bb)));
}

// =============================================================================
// Arithmetic operators for:
//
//     operator(Interval, numeric_type)
// =============================================================================

template <typename T, typename R>
Interval<T> operator+(const Interval<T>& left, const R& right)
{
    Interval<T> right_i = Interval<T>(right);
    return left + right_i;
}

template <typename T, typename R>
Interval<T> operator-(const Interval<T>& left, const R& right)
{
    Interval<T> right_i = Interval<T>(right);
    return left - right_i;
}

template <typename T, typename R>
Interval<T> operator*(const Interval<T>& left, const R& right)
{
    Interval<T> right_i = Interval<T>(right);
    return left * right_i;
}

template <typename T, typename R>
Interval<T> operator/(const Interval<T>& left, const R& right)
{
    Interval<T> right_i = Interval<T>(right);
    return left / right_i;
}

// =============================================================================
// Arithmetic operators for:
//
//     operator(numeric_type, Interval)
// =============================================================================

template <typename T, typename L>
Interval<T> operator+(const L& left, const Interval<T>& right)
{
    Interval<T> left_i = Interval<T>(left);
    return left_i + right;
}

template <typename T, typename L>
Interval<T> operator-(const L& left, const Interval<T>& right)
{
    Interval<T> left_i = Interval<T>(left);
    return left_i - right;
}

template <typename T, typename L>
Interval<T> operator*(const L& left, const Interval<T>& right)
{
    Interval<T> left_i = Interval<T>(left);
    return left_i * right;
}

template <typename T, typename L>
Interval<T> operator/(const L& left, const Interval<T>& right)
{
    Interval<T> left_i = Interval<T>(left);
    return left_i / right;
}
