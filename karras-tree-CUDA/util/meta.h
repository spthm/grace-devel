#pragma once

#include "../types.h"

#include "vector_types.h"

namespace grace {

// Usage:  typedef typename Real4ToRealMapper<Real4>::type Real
// Result: Real4 == float4  -> Real == float
//         Real4 == double4 -> Real == double
template <typename>
struct Real4ToRealMapper;

template <>
struct Real4ToRealMapper<float4> {
    typedef float type;
};

template <>
struct Real4ToRealMapper<double4> {
    typedef double type;
};


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


template <typename T>
struct VectorWord
{
    // float4 (alignment 16) is the largest type we can load with a single
    // instruction, or through texture fetches.
    typedef typename PredicateType<Divides<T, float4>::result, float4,
              typename PredicateType<Divides<T, float2>::result, float2,
                typename PredicateType<Divides<T, float>::result, float,
                  typename PredicateType<Divides<T, short>::result, short,
                    char>::type>::type>::type>::type type;
};

} // namespace grace