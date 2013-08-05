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

#include "../r3/vector.h"
#include "../r2/complex.h"
#include "../../util/optional.h"
#include <cmath>
#include <limits>

namespace s3d { namespace math { 

namespace detail
{
	template <class R, class Q1, class Q2>
	R do_quaternion_mul(const Q1 &q1, const Q2 &q2)/*{{{*/
	{
		// ... and Hamilton said: i^2 == j^2 == j^2 == ijk == -1
		return { + q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z,
				 + q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
				 + q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
				 + q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w };
	}/*}}}*/

	template <class R, class Q1, class Q2>
	R do_quaternion_div(const Q1 &q1, const Q2 &q2)/*{{{*/
	{
		auto den = sqrnorm(q2);

		return { (q2.w*q1.w + q2.x*q1.x + q2.y*q1.y + q2.z*q1.z)/den,
				 (q2.w*q1.x - q2.x*q1.w - q2.y*q1.z + q2.z*q1.y)/den,
				 (q2.w*q1.y + q2.x*q1.z + q2.y*q1.w - q2.z*q1.x)/den,
				 (q2.w*q1.z - q2.x*q1.y + q2.y*q1.x + q2.z*q1.z)/den };
	}/*}}}*/
}

template <class T, class U, template <class> class Q>
struct result_mul<r4::quaternion_like<Q<T>>, r4::quaternion_like<Q<U>>>/*{{{*/
{
	typedef Q<typename result_mul_dispatch<T,U>::type> type;

	static type call(const Q<T> &q1, const Q<U> &q2)
	{
		return detail::do_quaternion_mul<type>(q1, q2);
	}
};/*}}}*/

template <class T, class U, template <class> class Q>
struct result_div<r4::quaternion_like<Q<T>>, r4::quaternion_like<Q<U>>>/*{{{*/
{
	typedef Q<typename result_div_dispatch<T,U>::type> type;

	static type call(const Q<T> &q1, const Q<U> &q2)
	{
		return detail::do_quaternion_div<type>(q1, q2);
	}
};/*}}}*/

template <class T, class U>
struct result_mul<r4::quaternion_like<T>, r4::quaternion_like<U>>/*{{{*/
{
	typedef r4::Quaternion
	<
		typename result_mul_dispatch
		<
			typename value_type<T>::type,
			typename value_type<U>::type
		>::type
	> type;

	static type call(const T &q1, const U &q2)
	{
		return detail::do_quaternion_mul<type>(q1, q2);
	}
};/*}}}*/

template <class T, class U>
struct result_div<quaternion_like<T>, quaternion_like<U>>/*{{{*/
{
	typedef Quaternion
	<
		typename result_div_dispatch
		<
			typename value_type<T>::type,
			typename value_type<U>::type
		>::type
	> type;

	static type call(const T &q1, const U &q2)
	{
		return detail::do_quaternion_div<type>(q1, q2);
	}
};/*}}}*/

template <class T, class U>
struct result_add<quaternion_like<T>, arithmetic<U>>/*{{{*/
{
	typedef Quaternion
	<
		typename result_add_dispatch
		<
			typename value_type<T>::type, U
		>::type
	> type;
};/*}}}*/

template <class T, class U>
struct result_sub<quaternion_like<T>, arithmetic<U>>/*{{{*/
{
	typedef Quaternion
	<
		typename result_sub_dispatch
		<
			typename value_type<T>::type, U
		>::type
	> type;
};/*}}}*/

template <class T, class U>
struct result_mul<quaternion_like<T>, arithmetic<U>>/*{{{*/
{
	typedef Quaternion
	<
		typename result_mul_dispatch
		<
			typename value_type<T>::type, U
		>::type
	> type;
};/*}}}*/

template <class T, class U>
struct result_div<quaternion_like<T>, arithmetic<U>>/*{{{*/
{
	typedef Quaternion
	<
		typename result_div_dispatch
		<
			typename value_type<T>::type, U
		>::type
	> type;
};/*}}}*/


namespace r4
{

static_assert(order<quaternion>::value==1, "Wrong order for quaternion");
static_assert(std::is_same<value_type<quaternion>::type, real>::value,
			  "Wrong value_type for quaternion");

template <class Q> 
auto operator -(const Q &q)/*{{{*/
	-> typename requires<is_quaternion<Q>, Q>::type
{
	return { -q.w, -q.x, -q.y, -q.z };
}/*}}}*/
template <class Q1, class Q2> 
auto operator==(const Q1 &q1, const Q2 &q2)/*{{{*/
	-> typename requires<is_quaternion<Q1>, is_quaternion<Q2>, bool>::type
{
	return equal(q1.w,q2.w) && equal(q1.x,q2.x) && equal(q1.y,q2.y) && 
		   equal(q1.z,q2.z);
}/*}}}*/

template <class T> template <class Q, class>
auto Quaternion<T>::operator +=(const Q &that) -> Quaternion &/*{{{*/
{
	auto it2 = that.begin();
	for(auto it=this->begin(); it!=this->end(); ++it, ++it2)
		*it += *it2;
	return *this;
}/*}}}*/
template <class T> template <class Q, class>
auto Quaternion<T>::operator -=(const Q &that) -> Quaternion &/*{{{*/
{
	auto it2 = that.begin();
	for(auto it=this->begin(); it!=this->end(); ++it, ++it2)
		*it -= *it2;
	return *this;
}/*}}}*/

template <class T> template <class Q, class>
auto Quaternion<T>::operator *=(const Q &that) -> Quaternion &/*{{{*/
{
	return *this = *this * that;
}/*}}}*/
template <class T> template <class Q, class>
auto Quaternion<T>::operator /=(const Q &that) -> Quaternion &/*{{{*/
{
	return *this = *this / that;
}/*}}}*/

template <class T> 
Quaternion<T> &Quaternion<T>::operator +=(T d)/*{{{*/
{
	w += d;
	return *this;
}/*}}}*/
template <class T> 
Quaternion<T> &Quaternion<T>::operator -=(T d)/*{{{*/
{
	w -= d;
	return *this;
}/*}}}*/
template <class T> 
Quaternion<T> &Quaternion<T>::operator /=(T d)/*{{{*/
{
	for(auto it=this->begin(); it!=this->end(); ++it)
		*it /= d;
	return *this;
}/*}}}*/
template <class T> 
Quaternion<T> &Quaternion<T>::operator *=(T d)/*{{{*/
{
	for(auto it=this->begin(); it!=this->end(); ++it)
		*it *= d;
	return *this;
}/*}}}*/

template <class Q>
auto conj(const Q &q)/*{{{*/
	-> typename requires<is_quaternion<Q>, Q>::type
{
	return { q.w, -q.x, -q.y, -q.z };
}/*}}}*/
template <class T1, class T2, template<class> class Q1,template<class> class Q2>
auto dot(const Q1<T1> &q1, const Q2<T2> &q2)/*{{{*/
	-> typename requires<is_quaternion<Q1<T1>>, is_quaternion<Q2<T2>>,
				decltype(q1.x*q2.x)>::type
{
	return q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w;
}/*}}}*/
template <class T, template<class> class Q>
auto abs_imag(const Q<T> &q)/*{{{*/
	-> typename requires<is_quaternion<Q<T>>, T>::type
{
	return sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
}/*}}}*/

template <class T, template<class> class Q>
auto imag(const Q<T> &q)/*{{{*/
	-> typename requires<is_quaternion<Q<T>>, Vector<T,3>>::type
{
	return {q.x, q.y, q.z};
}/*}}}*/

template <class Q> 
auto sqr(const Q &q)/*{{{*/
	-> typename requires<is_quaternion<Q>, Q>::type
{
	return { q.w*q.w - q.x*q.x - q.y*q.y - q.z*q.z, 
		     2*q.w*q.x, 2*q.w*q.y, 2*q.w*q.z };
}/*}}}*/
template <class T, template<class> class Q> 
auto sqrt(const Q<T> &q)/*{{{*/
	-> typename requires<is_quaternion<Q<T>>, Q<T>>::type
{
	auto absim = abs_imag(q);
	auto z = sqrt(Complex<T>(q.w, absim));

	T m = absim==0 ? z.im : z.im / absim;

	return { z.re, m*q.x, m*q.y, m*q.z };
}/*}}}*/
template <class T, template<class> class Q> 
auto exp(const Q<T> &q)/*{{{*/
	-> typename requires<is_quaternion<Q<T>>, Quaternion<T>>::type
{
	auto absim = abs_imag(q);
	auto z = exp(Complex<T>(q.w, absim));

	T m = absim==0 ? z.im : z.im / absim;

	return { z.re, m*q.x, m*q.y, m*q.z };
}/*}}}*/
template <class T, template<class> class Q>
auto log(const Q<T> &q)/*{{{*/
	-> typename requires<is_quaternion<Q<T>>, Quaternion<T>>::type
{
	auto absim = abs_imag(q);
	auto z = log(Complex<T>(q.w, absim));

	T m = absim==0 ? z.im : z.im / absim;

	return { z.re, m*q.x, m*q.y, m*q.z };
}/*}}}*/
template <class T, template<class> class Q, class A> 
auto pow(const Q<T> &q, A x)/*{{{*/
	-> typename requires<is_quaternion<Q<T>>, Quaternion<T>>::type
{
	auto absim = abs_imag(q);
	auto z = pow(Complex<T>(q.w, absim), x);

	T m = absim==0 ? z.im : z.im / absim;

	return { z.re, m*q.x, m*q.y, m*q.z };
}/*}}}*/
template <class Q> 
auto inv(const Q &q)/*{{{*/
	-> typename requires<is_quaternion<Q>, Q>::type
{
	auto denom = sqrnorm(q);
	return { q.w/denom, -q.x/denom, -q.y/denom, -q.z/denom };
}/*}}}*/

template <class Q> 
auto operator<<(std::ostream &out, const Q &q)/*{{{*/
	-> typename requires<is_quaternion<Q>, std::ostream &>::type
{
	return out << "[" << q.w << "," << q.x << "," << q.y << "," << q.z << "]";
}/*}}}*/

}}} // namespace s3d::math::r4

// $Id: quaternion.hpp 3097 2010-09-02 00:58:35Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

