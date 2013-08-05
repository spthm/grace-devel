/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License version 3 as 
	published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public License
	along with S3D++.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef S3D_MATH_REAL
#define S3D_MATH_REAL

#include <cmath>
#include <cassert>
#include <limits>
#include <tuple>
#include <algorithm>
#include "../util/type_traits.h"
#include "../util/tuple.h" 
#include "../util/concepts.h"
#include "fwd.h"


namespace s3d { namespace math
{

using std::sin;
using std::cos;
using std::tan;
using std::atan2;
using std::asin;
using std::acos;
using std::abs;
using std::round;
using std::ceil;
using std::floor;
using std::log;
using std::exp;
using std::pow;
using std::isnan;
using std::isinf;
using std::sqrt;
using std::copysign;
using std::hypot;
using std::size_t;
using std::swap;
using std::swap_ranges;
using std::get;
using std::rint;

const real REAL_MAX = std::numeric_limits<real>::max();
const real EPSILON = 7e-6;

template <class T, class U>
bool equal(T a, U b, real eps=EPSILON)/*{{{*/
{
	assert(!isnan(a));
	assert(!isnan(b));

	typedef decltype(a-b) type;

	if(type(a) > type(b))
		return a-b < eps*2;
	else
		return b-a < eps*2;

	// não funciona com inteiros sem sinal
	//return abs(a-b) <= eps*2;
}/*}}}*/
template <class T, class U>
bool less_than(T a, U b, real eps=EPSILON)/*{{{*/
{
	assert(!isnan(a));
	assert(!isnan(b));

	return b-a > eps;
}/*}}}*/
template <class T, class U>
bool greater_or_equal_than(T a, U b, real eps=EPSILON)/*{{{*/
{
	return !less_than(a,b,eps);
}/*}}}*/
template <class T, class U>
bool greater_than(T a, U b, real eps=EPSILON)/*{{{*/
{
	return greater_or_equal_than(a,b,eps) && !equal(a,b,eps);
}/*}}}*/
template <class T, class U>
bool less_or_equal_than(T a, U b, real eps=EPSILON)/*{{{*/
{
	return less_than(a,b,eps) || equal(a,b,eps);
}/*}}}*/

template <class T, class U, class V> T clamp(T a, U min, V max)/*{{{*/
{
	if(a < T(min))
		return min;
	else if(a > T(max))
		return max;
	else
		return a;
}/*}}}*/

template <class T, class U>
inline T mod(T a, U _b)/*{{{*/
{
	T b = static_cast<T>(_b);
	assert(b != 0);

	int n = static_cast<int>((a+EPSILON)/b);
	a -= n*b;
	if(a < 0)
	{
		if(equal(a,0))
			a = 0;
		else
			a += b;
	}
	return a;
}/*}}}*/
template <class T, class U, class V>
inline T mod(T a, U _min, V _max)/*{{{*/
{
	T min = static_cast<T>(_min),
	  max = static_cast<T>(_max);

	return mod(a-min, max-min)+min;
}/*}}}*/

inline float rad(float a)/*{{{*/
{
	return a*(M_PI/180);
}/*}}}*/
inline double rad(double a)/*{{{*/
{
	return a*(M_PI/180);
}/*}}}*/
inline double rad(int a)/*{{{*/
{
	return a*(M_PI/180);
}/*}}}*/

inline double deg(int a)/*{{{*/
{
	return a/(M_PI/180);
}/*}}}*/
inline float deg(float a)/*{{{*/
{
	return a/(M_PI/180);
}/*}}}*/
inline double deg(double a)/*{{{*/
{
	return a/(M_PI/180);
}/*}}}*/

template <class T>
typename difference_type<T>::type deriv(const T &f0, const T &f1, real h)/*{{{*/
{
	return (f1 - f0)/h;
}/*}}}*/

template <class T, class U>
auto dot(const T &a, const U &b)/*{{{*/
	-> typename requires<std::is_arithmetic<T>, std::is_arithmetic<U>, 
						 decltype(a*b)>::type
{
	return a*b;
}/*}}}*/

template <class T> 
typename value_type<T>::type sqrnorm(const T &o)/*{{{*/
{
	return dot(o,o);
}/*}}}*/
template <class T> 
typename value_type<T>::type norm(const T &o)/*{{{*/
{
	return sqrt(sqrnorm(o));
}/*}}}*/
template <class T> 
typename value_type<T>::type abs(const T &o)/*{{{*/
{
	return norm(o);
}/*}}}*/
template <class T> 
bool is_unit(const T &o)/*{{{*/
{
	return equal(sqrnorm(o),1);
}/*}}}*/

template <class T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type area(const T &s)/*{{{*/
{
	return s;
}/*}}}*/

template <class T> 
auto map(real v)/*{{{*/
	-> typename std::enable_if<std::is_floating_point<T>::value, T>::type

{
	return v;
}/*}}}*/

template <class T> 
auto unmap(T v)/*{{{*/
	-> typename std::enable_if<std::is_floating_point<T>::value, real>::type

{
	return v;
}/*}}}*/

template <class T> 
auto map(real v)/*{{{*/
	-> typename std::enable_if<std::is_integral<T>::value, T>::type
{
	v = math::clamp(v, 0, 1);
	return std::min<real>(v*std::numeric_limits<T>::max(), 
						    std::numeric_limits<T>::max());
}/*}}}*/

template <class T> 
auto unmap(T v)/*{{{*/
	-> typename std::enable_if<std::is_integral<T>::value, real>::type
{
	return real(v) / std::numeric_limits<T>::max();
}/*}}}*/

template <class T>
T max(const T &head)
{
	return head;
}

template <class T, class... TT>
auto max(const T &head, const TT &...tail)
	-> typename std::common_type<T,TT...>::type 
{
	auto max_tail = max(tail...);
	return head > max_tail ? head : max_tail;
}

template <class T>
T min(const T &head)
{
	return head;
}

template <class T, class... TT>
auto min(const T &head, const TT &...tail)
	-> typename std::common_type<T,TT...>::type 
{
	auto min_tail = min(tail...);
	return head < min_tail ? head : min_tail;
}

#if 0 // BAD TRIP
// floating point
template <class T> 
auto clamp_add(T a, T b)/*{{{*/
	-> typename requires<std::is_floating_point<T>, T>::type
{
	return a + b;
}/*}}}*/

template <class T> 
auto clamp_sub(T a, T b)/*{{{*/
	-> typename requires<std::is_floating_point<T>, T>::type
{
	return a - b;
}/*}}}*/

template <class T> 
auto clamp_mul(T a, T b)/*{{{*/
	-> typename requires<std::is_floating_point<T>, T>::type
{
	return a * b;
}/*}}}*/

// integral
template <class T> 
auto clamp_add(T a, T b)/*{{{*/
	-> typename requires<std::is_integral<T>, T>::type
{
	typedef typename higher_precision<T>::type higher;

	return clamp(higher(a) + higher(b), 
				 std::numeric_limits<T>::min(), 
				 std::numeric_limits<T>::max());
}/*}}}*/

template <class T> 
auto clamp_sub(T a, T b)/*{{{*/
	-> typename requires<std::is_integral<T>, T>::type
{
	typedef typename std::make_signed<typename higher_precision<T>::type>::type higher;

	return clamp(higher(a) - higher(b), 
				 std::numeric_limits<T>::min(), 
				 std::numeric_limits<T>::max());
}/*}}}*/

template <class T> 
auto clamp_mul(T a, T b)/*{{{*/
	-> typename requires<std::is_integral<T>, T>::type
{
	typedef typename higher_precision<T>::type higher;

	return clamp(higher(a) * higher(b), 
				 std::numeric_limits<T>::min(), 
				 std::numeric_limits<T>::max());
}/*}}}*/

// integral & floating point
template <class T> 
auto clamp_div(T a, T b) -> T/*{{{*/
{
	return a / b;
}/*}}}*/

#endif

} // namespace math

using math::REAL_MAX;
using math::equal;
using math::less_than;
using math::greater_or_equal_than;
using math::greater_than;
using math::less_or_equal_than;
using math::max;
using math::min;
using math::deg;
using math::rad;

} // namespace s3d

#endif

// $Id: real.h 3115 2010-09-06 17:41:29Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

