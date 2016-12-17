#pragma once

// No grace/vector.h include.
// This should only ever be included by vector.h.
#include "grace/generic/functional.h"

#include <algorithm>
#include <cmath>

namespace grace {

namespace detail {

//
// op(v)
//

template <size_t Dims, typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<Dims, T> operator_loop(const Vector<Dims, T>& v,
                              const Operator& op)
{
    Vector<Dims, T> result;

#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (size_t i = 0; i < Dims; ++i) {
        result[i] = op(v[i]);
    }

    return result;
}

// Overload for Vector<3, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<3, T> operator_loop(const Vector<3, T>& v,
                           const Operator& op)
{
    Vector<3, T> result;

    result.x = op(v.x);
    result.y = op(v.y);
    result.z = op(v.z);

    return result;
}

// Overload for Vector<4, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<4, T> operator_loop(const Vector<4, T>& v,
                           const Operator& op)
{
    Vector<4, T> result;

    result.x = op(v.x);
    result.y = op(v.y);
    result.z = op(v.z);
    result.w = op(v.w);

    return result;
}


//
// u op v
//

template <size_t Dims, typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<Dims, T> operator_loop(const Vector<Dims, T>& u,
                              const Vector<Dims, T>& v,
                              const Operator& op)
{
    Vector<Dims, T> result;

#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (size_t i = 0; i < Dims; ++i) {
        result[i] = op(u[i], v[i]);
    }

    return result;
}

// Overload for Vector<3, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<3, T> operator_loop(const Vector<3, T>& u,
                           const Vector<3, T>& v,
                           const Operator& op)
{
    Vector<3, T> result;

    result.x = op(u.x, v.x);
    result.y = op(u.y, v.y);
    result.z = op(u.z, v.z);

    return result;
}

// Overload for Vector<4, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<4, T> operator_loop(const Vector<4, T>& u,
                           const Vector<4, T>& v,
                           const Operator& op)
{
    Vector<4, T> result;

    result.x = op(u.x, v.x);
    result.y = op(u.y, v.y);
    result.z = op(u.z, v.z);
    result.w = op(u.w, v.w);

    return result;
}


//
// v op scalar
//

template <size_t Dims, typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<Dims, T> operator_loop(const Vector<Dims, T>& v,
                              const T s,
                              const Operator& op)
{
    Vector<Dims, T> result;

#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (size_t i = 0; i < Dims; ++i) {
        result[i] = op(v[i], s);
    }

    return result;
}

// Overload for Vector<3, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<3, T> operator_loop(const Vector<3, T>& v,
                           const T s,
                           const Operator& op)
{
    Vector<3, T> result;

    result.x = op(v.x, s);
    result.y = op(v.y, s);
    result.z = op(v.z, s);

    return result;
}

// Overload for Vector<4, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<4, T> operator_loop(const Vector<4, T>& v,
                           const T s,
                           const Operator& op)
{
    Vector<4, T> result;

    result.x = op(v.x, s);
    result.y = op(v.y, s);
    result.z = op(v.z, s);
    result.w = op(v.w, s);

    return result;
}


//
// scalar op v
//

template <size_t Dims, typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<Dims, T> operator_loop(const T s,
                              const Vector<Dims, T>& v,
                              const Operator& op)
{
    Vector<Dims, T> result;

#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (size_t i = 0; i < Dims; ++i) {
        result[i] = op(s, v[i]);
    }

    return result;
}

// Overload for Vector<3, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<3, T> operator_loop(const T s,
                           const Vector<3, T>& v,
                           const Operator& op)
{
    Vector<3, T> result;

    result.x = op(s, v.x);
    result.y = op(s, v.y);
    result.z = op(s, v.z);

    return result;
}

// Overload for Vector<4, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
Vector<4, T> operator_loop(const T s,
                           const Vector<4, T>& v,
                           const Operator& op)
{
    Vector<4, T> result;

    result.x = op(s, v.x);
    result.y = op(s, v.y);
    result.z = op(s, v.z);
    result.w = op(s, v.w);

    return result;
}


//
// reduce(v, op)
//

template <size_t Dims, typename T, typename Operator>
GRACE_HOST_DEVICE
T operator_reduce(const Vector<Dims, T>& v,
                  const Operator& op, const T init)
{
    T result(init);

#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (size_t i = 0; i < Dims; ++i) {
        result = op(result, v[i]);
    }

    return result;
}

// Overload for Vector<3, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
T operator_reduce(const Vector<3, T>& v,
                  const Operator& op, const T init)
{
    T result(init);

    result = op(result, v.x);
    result = op(result, v.y);
    result = op(result, v.z);

    return result;
}

// Overload for Vector<4, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
T operator_reduce(const Vector<4, T>& v,
                  const Operator& op, const T init)
{
    T result(init);

    result = op(result, v.x);
    result = op(result, v.y);
    result = op(result, v.z);
    result = op(result, v.w);

    return result;
}


//
// reduce(op_inner(u, v), op_outer)
//

template <typename OutType, size_t Dims, typename InType,
          typename OperatorInner, typename OperatorOuter>
GRACE_HOST_DEVICE
OutType operator_reduce(const Vector<Dims, InType>& u,
                        const Vector<Dims, InType>& v,
                        const OperatorInner& op_inner,
                        const OperatorOuter& op_outer,
                        const OutType outer_init)
{
    OutType result(outer_init);

#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (size_t i = 0; i < Dims; ++i) {
        result = op_outer(result, op_inner(u[i], v[i]));
    }

    return result;
}

// Overload for Vector<3, > to avoid potentially-unsafe array-accessor.
template <typename OutType, typename InType,
          typename OperatorInner, typename OperatorOuter>
GRACE_HOST_DEVICE
OutType operator_reduce(const Vector<3, InType>& u,
                        const Vector<3, InType>& v,
                        const OperatorInner& op_inner,
                        const OperatorOuter& op_outer,
                        const OutType outer_init)
{
    OutType result(outer_init);

    result = op_outer(result, op_inner(u.x, v.x));
    result = op_outer(result, op_inner(u.y, v.y));
    result = op_outer(result, op_inner(u.z, v.z));

    return result;
}

// Overload for Vector<4, > to avoid potentially-unsafe array-accessor.
template <typename OutType, typename InType,
          typename OperatorInner, typename OperatorOuter>
GRACE_HOST_DEVICE
OutType operator_reduce(const Vector<4, InType>& u,
                        const Vector<4, InType>& v,
                        const OperatorInner& op_inner,
                        const OperatorOuter& op_outer,
                        const OutType outer_init)
{
    OutType result(outer_init);

    result = op_outer(result, op_inner(u.x, v.x));
    result = op_outer(result, op_inner(u.y, v.y));
    result = op_outer(result, op_inner(u.z, v.z));
    result = op_outer(result, op_inner(u.w, v.w));

    return result;
}

} // namespace detail


//
// Comparison operations
//

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
bool operator==(const Vector<Dims, T>& lhs, const Vector<Dims, T>& rhs)
{
    return detail::operator_reduce<bool>(lhs, rhs,
                                         equal_to<T>(), logical_and<bool>(),
                                         true);
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
bool operator!=(const Vector<Dims, T>& lhs, const Vector<Dims, T>& rhs)
{
    return detail::operator_reduce<bool>(lhs, rhs,
                                         not_equal_to<T>(), logical_or<bool>(),
                                         false);
}


//
// Vector operations
//

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const Vector<Dims, T>& vec)
{
    return detail::operator_loop(vec, negate<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T max_element(const Vector<Dims, T>& vec)
{
    return detail::operator_reduce(vec, maximum<T>(), vec[0]);
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T min_element(const Vector<Dims, T>& vec)
{
    return detail::operator_reduce(vec, minimum<T>(), vec[0]);
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T norm2(const Vector<Dims, T>& vec)
{
    return dot(vec, vec);
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T norm(const Vector<Dims, T>& vec)
{
#ifdef __CUDA_ARCH__
    return sqrt(norm2(vec));
#else
    return std::sqrt(norm2(vec));
#endif
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> normalize(const Vector<Dims, T>& vec)
{
#ifdef __CUDA_ARCH__
    T s = rsqrt(norm2(vec));
#else
    T s = 1.0 / std::sqrt(norm2(vec));
#endif

    return s * vec;
}


//
// Vector-vector operations
//

// Cross-product only definied for length-three vectors.
template <typename T>
GRACE_HOST_DEVICE
Vector<3, T> cross(const Vector<3, T>& u, const Vector<3, T>& v)
{
    Vector<3, T> result;

    result.x = u.y * v.z - u.z * v.y;
    result.y = u.z * v.x - u.x * v.z;
    result.z = u.x * v.y - u.y * v.x;

    return result;
}

// Angular separation only defined for length-three (and not defined length-two)
// vectors
template <typename T>
GRACE_HOST_DEVICE
T angular_separation(const Vector<3, T>& u, const Vector<3, T>& v)
{
    // This form is well conditioned when the angle is 0 or pi; acos is not.
    return atan2( norm(cross(u, v)), dot(u, v) );
}

// Great circle distance only defined for length-three vectors.
template <typename T>
GRACE_HOST_DEVICE
T great_circle_distance(const Vector<3, T>& u, const Vector<3, T>& v,
                        const T radius)
{
    return radius * angular_separation(u, v);
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T dot(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_reduce(u, v, multiplies<T>(), plus<T>(), (T)0);
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> max(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, maximum<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> min(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, minimum<T>());
}


//
// Vector-vector arithmetic
//

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator+(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, plus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, minus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator*(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, multiplies<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator/(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, divides<T>());
}


///
// Vector-scalar arithmetic
//

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator+(const Vector<Dims, T>& v, const T s)
{
    return detail::operator_loop(v, s, plus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const Vector<Dims, T>& v, const T s)
{
    return detail::operator_loop(v, s, minus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator*(const Vector<Dims, T>& v, const T s)
{
    return detail::operator_loop(v, s, multiplies<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator/(const Vector<Dims, T>& v, const T s)
{
    return detail::operator_loop(v, s, divides<T>());
}


//
// Scalar-vector arithmetic
//

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator+(const T s, const Vector<Dims, T>& v)
{
    return detail::operator_loop(s, v, plus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const T s, const Vector<Dims, T>& v)
{
    return detail::operator_loop(s, v, minus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator*(const T s, const Vector<Dims, T>& v)
{
    return detail::operator_loop(s, v, multiplies<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator/(const T s, const Vector<Dims, T>& v)
{
    return detail::operator_loop(s, v, divides<T>());
}

} // namespace grace
