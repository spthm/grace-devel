#pragma once

#include "grace/config.h"

namespace grace {

template <typename T, typename U>
struct are_same
{
    static const bool result = false;
};

template <typename T>
struct are_same<T, T>
{
    static const bool result = true;
};


template <typename T, typename U>
GRACE_HOST_DEVICE bool are_types_equal() {
    return are_same<T, U>::result;
}

// U may be deduced.
template <typename T, typename U>
GRACE_HOST_DEVICE bool are_types_equal(const U value) {
    return are_same<T, U>::result;
}

// T and U may be deduced.
template <typename T, typename U>
GRACE_HOST_DEVICE bool are_types_equal(const T Tvalue, const U Uvalue) {
    return are_same<T, U>::result;
}


template <bool Predicate, typename TrueType, typename FalseType>
struct PredicateType
{
    typedef TrueType type;
};

template <typename TrueType, typename FalseType>
struct PredicateType<false, TrueType, FalseType>
{
    typedef FalseType type;
};


template <typename T, typename Divisor>
struct Divides
{
    static const bool result = (sizeof(T) % sizeof(Divisor) == 0);
};

} // namespace grace
