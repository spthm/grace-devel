#pragma once

#include "grace/vector.h"

#ifdef __CUDACC__
#include <math.h>
#else
#include <cmath>
#endif

#include <algorithm>
#include <functional>

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
    for (size_t i = 0; i < Dims, ++i) {
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
    resuly.y = op(v.y);
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
    resuly.y = op(v.y);
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
    for (size_t i = 0; i < Dims, ++i) {
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
    resuly.y = op(u.y, v.y);
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
    resuly.y = op(u.y, v.y);
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
    for (size_t i = 0; i < Dims, ++i) {
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
    resuly.y = op(v.y, s);
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
    resuly.y = op(v.y, s);
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
    for (size_t i = 0; i < Dims, ++i) {
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
    resuly.y = op(s, v.y);
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
    resuly.y = op(s, v.y);
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
    for (size_t i = 0; i < Dims, ++i) {
        result = op(result, v[i])
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

template <size_t Dims, typename T,
          typename OperatorInner, typename OperatorOuter>
GRACE_HOST_DEVICE
Toperator_reduce(const Vector<Dims, T>& u,
                 const Vector<Dims, T>& v,
                 const OperatorInner& op_inner,
                 const OperatorOuter& op_outer,
                 const T outer_init)
{
    T result(outer_init);

#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (size_t i = 0; i < Dims, ++i) {
        result = op_outer(result, op_inner(u[i], v[i]));
    }

    return result;
}

// Overload for Vector<3, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
T operator_reduce(const Vector<3, T>& v,
                  const Vector<3, T>& u,
                  const OperatorInner& op_inner,
                  const OperatorOuter& op_outer,
                  const T outer_init)
{
    T result(outer_init);

    result = op_outer(result, op_inner(u.x, v.x));
    result = op_outer(result, op_inner(u.y, v.y));
    result = op_outer(result, op_inner(u.z, v.z));

    return result;
}

// Overload for Vector<4, > to avoid potentially-unsafe array-accessor.
template <typename T, typename Operator>
GRACE_HOST_DEVICE
T operator_reduce(const Vector<4, T>& v,
                  const Vector<4, T>& u,
                  const OperatorInner& op_inner,
                  const OperatorOuter& op_outer,
                  const T outer_init)
{
    T result(outer_init);

    result = op_outer(result, op_inner(u.x, v.x));
    result = op_outer(result, op_inner(u.y, v.y));
    result = op_outer(result, op_inner(u.z, v.z));
    result = op_outer(result, op_inner(u.w, v.w));

    return result;
}

} // namespace detail


/* Vector opeartions */

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const Vector<Dims, T>& vec)
{
    return detail::operator_loop(vec, std::negate<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T max_element(const Vector<Dims, T>& vec)
{
    return operator_reduce(vec, std::max<T>(), vec[0]);
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
T min_element(const Vector<Dims, T>& vec)
{
    return operator_reduce(vec, std::min<T>(), vec[0]);
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
#ifdef __CUDACC__
    return sqrt(norm2(vec));
#else
    return std::sqrt(norm2(vec));
#endif
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> normalize(const Vector<Dims, T>& vec)
{
#ifdef __CUDACC__
    T s = rsqrt(norm2(vec));
#else
    T s = 1.0 / std::sqrt(norm2(vec));
#endif

    return s * vec;
}


/* Vector-vector operations */

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

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> dot(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_reduce(u, v, std::multiplies<T>(), std::plus<T>(),
                                   (T)0);
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> max(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, std::max<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> min(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, std::min<T>());
}


/* Vector-vector arithmetic */

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator+(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, std:plus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, std::minus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator*(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, std::multiplies<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator/(const Vector<Dims, T>& u, const Vector<Dims, T>& v)
{
    return detail::operator_loop(u, v, std::divides<T>());
}


/* Vector-scalar aarithmetic */

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator+(const Vector<Dims, T>& v, const T s)
{
    return detail::operator_loop(v, s, std::plus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const Vector<Dims, T>& v, const T s)
{
    return detail::operator_loop(v, s, std::minus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator*(const Vector<Dims, T>& v, const T s)
{
    return detail::operator_loop(v, s, std::multiplies<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator/(const Vector<Dims, T>& v, const T s)
{
    return detail::operator_loop(v, s, std::divides<T>());
}


/* Scalar-vector arithmetic */

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator+(const T s, const Vector<Dims, T>& v)
{
    return detail::operator_loop(s, v, std::plus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator-(const T s, const Vector<Dims, T>& v)
{
    return detail::operator_loop(s, v, std::minus<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator*(const T s, const Vector<Dims, T>& v)
{
    return detail::operator_loop(s, v, std::multiplies<T>());
}

template <size_t Dims, typename T>
GRACE_HOST_DEVICE
Vector<Dims, T> operator/(const T s, const Vector<Dims, T>& v)
{
    return detail::operator_loop(s, v, std::divides<T>());
}

} // namespace grace
