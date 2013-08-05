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

#include "../real.h"

namespace s3d { namespace math 
{

namespace r2
{

static_assert(std::is_same<value_type<Complex<double>>::type, double>::value,
			  "Bad value_type for complex");

template <class C> 
auto operator -(const C &c)/*{{{*/
	-> typename requires<is_complex<C>, C>::type
{
	return { -c.re, -c.im };
}/*}}}*/

// Arithmetics Complex x Complex

template <class C1, class C2>
auto operator +=(C1 &&c1, const C2 &c2)/*{{{*/
	->  typename std::enable_if<is_complex<C1>::value && 
								is_complex<C2>::value, C1&&>::type
{
	auto it1 = c1.begin(); auto it2 = c2.begin();
	while(it1 != c1.end())
		*it1++ += *it2++;
	return std::forward<C1>(c1);
}/*}}}*/

template <class C1, class C2>
auto operator -=(C1 &&c1, const C2 &c2)/*{{{*/
	->  typename std::enable_if<is_complex<C1>::value && 
								is_complex<C2>::value, C1&&>::type
{
	auto it1 = c1.begin(); auto it2 = c2.begin();
	while(it1 != c1.end())
		*it1++ -= *it2++;
	return std::forward<C1>(c1);
}/*}}}*/

template <class C1, class C2>
auto operator *=(C1 &&c1, const C2 &c2)/*{{{*/
	->  typename std::enable_if<is_complex<C1>::value && 
								is_complex<C2>::value, C1&&>::type
{
	auto c1_re = c1.re;
	c1.re = c1.re*c2.re - c1.im*c2.im;
	c1.im = c1.im*c2.re + c1_re*c2.im;

	return std::forward<C1>(c1);
}/*}}}*/

template <class C1, class C2>
auto operator /=(C1 &&c1, const C2 &c2)/*{{{*/
	->  typename std::enable_if<is_complex<C1>::value && 
								is_complex<C2>::value, C1&&>::type
{
	c1 *= conj(c2);
	c1 /= sqrnorm(c2);
	return std::forward<C1>(c1);
}/*}}}*/

// Arithmetics Complex x Real

template <class C>
auto operator +=(C &&c, typename value_type<C>::type d)/*{{{*/
	-> typename std::enable_if<is_complex<C>::value, C&&>::type
{
	c.re += d;
	return std::forward<C>(c);
}/*}}}*/

template <class C>
auto operator -=(C &&c, typename value_type<C>::type d)/*{{{*/
	-> typename std::enable_if<is_complex<C>::value, C &&>::type
{
	c.re -= d;
	return std::forward<C>(c);
}/*}}}*/

template <class C>
auto operator *=(C &&c, typename value_type<C>::type d)/*{{{*/
	-> typename std::enable_if<is_complex<C>::value, C&&>::type
{
	for(auto it=c.begin(); it!=c.end(); ++it)
		*it *= d;
	return std::forward<C>(c);
}/*}}}*/

template <class C>
auto operator /=(C &&c, typename value_type<C>::type d)/*{{{*/
	-> typename std::enable_if<is_complex<C>::value, C&&>::type
{
	for(auto it=c.begin(); it!=c.end(); ++it)
		*it /= d;
	return std::forward<C>(c);
}/*}}}*/

template <class C1, class C2> 
auto operator==(const C1 &c1, const C2 &c2)/*{{{*/
	-> typename requires<is_complex<C1>, is_complex<C2>, bool>::type
{
	return equal(c1.re,c2.re) && equal(c1.im,c2.im);
}/*}}}*/

template <class C1, class C2>
auto dot(const C1 &c1, const C2 &c2)/*{{{*/
	-> typename requires<is_complex<C1>, is_complex<C2>,
	         decltype(c1.re*c2.re)>::type
{
	return c1.re*c2.re + c1.im*c2.im;
}/*}}}*/

using math::sqrt;
using math::log;
using math::exp;
using math::sin;
using math::cos;
using math::tan;

template <class C> auto conj(const C &c)/*{{{*/
	-> typename requires<is_complex<C>, C>::type
{
	return { c.re, -c.im };
}/*}}}*/

template <class C> auto sqr(const C &c)/*{{{*/
	-> typename requires<is_complex<C>, C>::type
{
	return {c.re*c.re - c.im*c.im, 2*c.re*c.im};
}/*}}}*/

template <class C> auto sqrt(const C &c)/*{{{*/
	-> typename requires<is_complex<C>, C>::type
{
	auto a = abs(c);
	auto m = c.re >= 0 ? 0.5 : -0.5;

	return {sqrt(2*(a+c.re)), m*sqrt(2*(a-c.re))};
}/*}}}*/

template <class T> Complex<T> exp(const Complex<T> &c)/*{{{*/
{
	if(c.im == 0)
		return Complex<T>(exp(c.re), 0);
	else if(c.re == 0)
		return Complex<T>(cos(c.im), sin(c.im));
	else
	{
		T e = exp(c.re);
		return Complex<T>(e*cos(c.im), e*sin(c.im));
	}
}/*}}}*/
template <class T> Complex<T> log(const Complex<T> &c)/*{{{*/
{
	if(c.im == 0)
	{
		if(c.re >= 0)
			return Complex<T>(log(c.re), 0);
		else 
			return Complex<T>(log(-c.re), M_PI);
	}
	else if(c.re == 0)
	{
		if(c.im > 0)
			return Complex<T>(log(c.im), M_PI/2);
		else
			return Complex<T>(log(-c.im), -M_PI/2);
	}
	else
		return Complex<T>(0.5*log(sqrnorm(c)), atan2(c.im, c.re));
}/*}}}*/
template <class T> Complex<T> sin(const Complex<T> &c)/*{{{*/
{
	Complex<T> iz = Complex<T>(0,1)*c;
	return (exp(iz) - exp(-iz)) / Complex<T>(0,2);
}/*}}}*/
template <class T> Complex<T> cos(const Complex<T> &c)/*{{{*/
{
	Complex<T> iz = Complex<T>(0,1)*c;
	return (exp(iz) + exp(-iz)) / 2;
}/*}}}*/
template <class T> Complex<T> tan(const Complex<T> &c)/*{{{*/
{
	return sin(c)/cos(c);
}/*}}}*/

}}} // namespace s3d::math::r2

// $Id: complex.hpp 2971 2010-08-17 23:43:09Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

