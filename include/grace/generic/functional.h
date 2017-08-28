#pragma once

#include "grace/config.h"

#include <functional>

/* This file provides device-compatible implementations of most of the standard
 * library header <functional>. This allows generic host- and device-compatible
 * code to be written that will compile (for the host) when CUDA/Thrust is not
 * available.
 */

namespace grace {

//
// Arithmetic
//

template <typename T>
struct plus : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs + rhs;
    }
};

template <typename T>
struct minus : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs - rhs;
    }
};

template <typename T>
struct multiplies : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs * rhs;
    }
};

template <typename T>
struct divides : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs / rhs;
    }
};

template <typename T>
struct modulus : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs % rhs;
    }
};


template <typename T>
struct maximum : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs < rhs ? rhs : lhs;
    }
};

template <typename T>
struct minimum : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }
};

template <typename T>
struct negate : std::unary_function<T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& x) const
    {
        return -x;
    }
};


//
// Comparisons
//

template <typename T>
struct equal_to : std::binary_function<T, T, bool>
{
    GRACE_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs == rhs;
    }
};

template <typename T>
struct not_equal_to : std::binary_function<T, T, bool>
{
    GRACE_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs != rhs;
    }
};

template <typename T>
struct greater : std::binary_function<T, T, bool>
{
    GRACE_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs > rhs;
    }
};

template <typename T>
struct less : std::binary_function<T, T, bool>
{
    GRACE_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs < rhs;
    }
};

template <typename T>
struct greater_equal : std::binary_function<T, T, bool>
{
    GRACE_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs >= rhs;
    }
};

template <typename T>
struct less_equal : std::binary_function<T, T, bool>
{
    GRACE_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs <= rhs;
    }
};


//
// Logical operations
//

template <typename T>
struct logical_and : std::binary_function<T, T, bool>
{
    GRACE_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs && rhs;
    }
};

template <typename T>
struct logical_or : std::binary_function<T, T, bool>
{
    GRACE_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs || rhs;
    }
};

template <typename T>
struct logical_not : std::unary_function<T, bool>
{
    GRACE_HOST_DEVICE bool operator()(const T& x) const
    {
        return !x;
    }
};


//
// Bitwise operations
//

template <typename T>
struct bit_and : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs & rhs;
    }
};

template <typename T>
struct bit_or : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs | rhs;
    }
};

template <typename T>
struct bit_xor : std::binary_function<T, T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& lhs, const T& rhs) const
    {
        return lhs ^ rhs;
    }
};

template <typename T>
struct bit_not : std::unary_function<T, T>
{
    GRACE_HOST_DEVICE T operator()(const T& x) const
    {
        return ~x;
    }
};


//
// Negators
//

template<typename Predicate>
struct unary_negate : std::unary_function<typename Predicate::argument_type,
                                          bool>
{
    GRACE_HOST_DEVICE explicit unary_negate(const Predicate& pred)
        : pred(pred) {}

    GRACE_HOST_DEVICE
    bool operator()(const typename Predicate::argument_type& x) const
    {
        return !pred(x);
    }

protected:
    Predicate pred;
};

template<typename Predicate>
struct binary_negate
    : std::binary_function<typename Predicate::first_argument_type,
                           typename Predicate::second_argument_type,
                           bool>
{
    GRACE_HOST_DEVICE explicit binary_negate(const Predicate& pred)
        : pred(pred) {}

    GRACE_HOST_DEVICE
    bool operator()(const typename Predicate::first_argument_type& x,
                    const typename Predicate::second_argument_type& y) const
    {
        return !pred(x, y);
    }

protected:
    Predicate pred;
};

} // namespace grace
