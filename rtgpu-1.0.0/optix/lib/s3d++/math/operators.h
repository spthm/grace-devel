#ifndef S3D_MATH_OPERATORS_H
#define S3D_MATH_OPERATORS_H

namespace s3d { namespace math
{

// to be specialized by user types
template <class V1, class V2> struct result_add;
template <class V1, class V2> struct result_mul;

template <class V1, class V2> struct result_sub;
template <class V1, class V2> struct result_div;

// dispatcher to call the right result_*
template <class V1, class V2> struct result_add_dispatch;
template <class V1, class V2> struct result_mul_dispatch;

template <class V1, class V2> struct result_sub_dispatch;
template <class V1, class V2> struct result_div_dispatch;

struct operators {}; // derive from it to enable our operators to kick in

} // namespace math

} // namespace s3d::math

#include "operators.hpp"

#endif
