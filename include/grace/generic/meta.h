#pragma once

#include "grace/types.h"

namespace grace {

// Usage:  typedef typename Real{2,3,4}ToRealMapper<Real{2,3,4}>::type Real
// Result: Real{2,3,4} == float{2,3,4}  -> Real == float
//         Real{2,3,4} == double{2,3,4} -> Real == double
template <typename>
struct Real2ToRealMapper;

template <>
struct Real2ToRealMapper<float2> {
    typedef float type;
};

template <>
struct Real2ToRealMapper<double2> {
    typedef double type;
};

template <typename>
struct Real3ToRealMapper;

template <>
struct Real3ToRealMapper<float3> {
    typedef float type;
};

template <>
struct Real3ToRealMapper<double3> {
    typedef double type;
};

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

// Usage:  typedef typename RealToReal{2,3,4}Mapper<Real>::type Real{2,3,4}
// Result: Real == float  -> Real{2,3,4} == float{2,3,4}
//         Real == double -> Real{2,3,4} == double{2,3,4}
template <typename>
struct RealToReal2Mapper;

template <>
struct RealToReal2Mapper<float> {
    typedef float2 type;
};

template <>
struct RealToReal2Mapper<double> {
    typedef double2 type;
};

template <typename>
struct RealToReal3Mapper;

template <>
struct RealToReal3Mapper<float> {
    typedef float3 type;
};

template <>
struct RealToReal3Mapper<double> {
    typedef double3 type;
};

template <typename>
struct RealToReal4Mapper;

template <>
struct RealToReal4Mapper<float> {
    typedef float4 type;
};

template <>
struct RealToReal4Mapper<double> {
    typedef double4 type;
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
