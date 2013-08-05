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

#include <cassert>
#include <cmath>
#include <limits>
#include "vector.h"

namespace s3d  {

namespace math 
{

static_assert(order<Polar<int,double>>::value==1, "Error in polar order");
static_assert(std::is_same<value_type<Polar<double,float>>::type,
						   double>::value, "Error in polar value_type");

template <class T, class A, class U, class B>
struct result_add<Polar<T,A>,Polar<U,B>>
{
	typedef decltype(cos(std::declval<A>()+std::declval<B>())) angle_type;

	typedef Polar<decltype(sqrt(std::declval<T>()+std::declval<U>()
								+std::declval<angle_type>())),
			      decltype(acos(std::declval<T>()+std::declval<U>()*
			      				std::declval<angle_type>()))> type;
};

template <class T, class A, class U, class B>
struct result_sub<Polar<T,A>,Polar<U,B>> : result_add<Polar<T,A>,Polar<U,B>> {};

template <class T, class A, class U, class B>
struct result_mul<Polar<T,A>,Polar<U,B>>
{
	typedef Polar<typename result_mul<T,U>::type,
			      typename result_add<A,B>::type> type;
};

template <class T, class A, class U, class B>
struct result_div<Polar<T,A>,Polar<U,B>>
{
	typedef Polar<typename result_div<T,U>::type,
			      typename result_sub<A,B>::type> type;
};

namespace r2
{
template <class T, class A>
Polar<T,A>::Polar(const Vector<T,2> &cart)/*{{{*/
	: coords_base(norm(cart), angle<A>(cart))
{
}/*}}}*/

template <class T, class A>
Polar<T,A>::operator Vector<T,2>() const/*{{{*/
{
	return Vector<T,2>(r*cos(theta), r*sin(theta));
}/*}}}*/

template <class T, class A> 
bool Polar<T,A>::operator==(const Polar &that) const/*{{{*/
{
	auto p1 = normalize(*this),
		 p2 = normalize(that);

	if(!equal(p1.r,p2.r))
		return false;
	else if(equal(p1.r,0))
		return true;

	return equal(p1.theta, p2.theta);
}/*}}}*/

template <class T, class A> 
Polar<T,A> Polar<T,A>::operator -() const/*{{{*/
{
	return {-r, theta};
}/*}}}*/

template <class T, class A> template <class U, class A2>
Polar<T,A> &Polar<T,A>::operator +=(const Polar<U,A2> &v)/*{{{*/
{
	auto cdif = cos(theta - v.theta),
	     newr = sqrt(r*r + v.r*v.r + 2*r*v.r*cdif);

	if(newr > 0)
		theta += acos((r + v.r*cdif)/newr);

	r = newr;

	return *this;
}/*}}}*/
template <class T, class A> template <class U, class A2>
Polar<T,A> &Polar<T,A>::operator -=(const Polar<U,A2> &v)/*{{{*/
{
	return *this += -v;
}/*}}}*/
template <class T, class A> template <class U, class A2>
Polar<T,A> &Polar<T,A>::operator *=(const Polar<U,A2> &v)/*{{{*/
{
	r *= v.r;
	theta += v.theta;
	return *this;
}/*}}}*/
template <class T, class A> template <class U, class A2>
Polar<T,A> &Polar<T,A>::operator /=(const Polar<U,A2> &v)/*{{{*/
{
	r /= v.r;
	theta -= v.theta;
	return *this;
}/*}}}*/

template <class T, class A> 
Polar<T,A> &Polar<T,A>::operator *=(T d)/*{{{*/
{
	r *= d;
	return *this;
}/*}}}*/
template <class T, class A> 
Polar<T,A> &Polar<T,A>::operator /=(T d)/*{{{*/
{
	r /= d;
	return *this;
}/*}}}*/

template <class T, class A> 
std::ostream &operator<<(std::ostream &out, const Polar<T,A> &v)/*{{{*/
{
	return out << '[' << (equal(v.r,0)?0:v.r) 
		       << '<' << (equal(v.theta,0)?0:deg(v.theta)) << "°"
		       << ']';
}/*}}}*/
template <class T, class A> 
std::istream &operator>>(std::istream &in, Polar<T,A> &v)/*{{{*/
{
	if(in.peek() == '[')
		in.ignore();

	Polar<T,A> aux;

	in >> aux.r;
	if(in.peek() == '<')
		in.ignore();

	in >> aux.theta;

	if(!in)
		return in;

	if(in.peek() == 'o')
	{
		aux.theta = rad(aux.theta);
		in.ignore();
	}

	if(!in.eof() && in.peek() == ']')
		in.ignore();

	if(in)
		v = aux;

	return in;
}/*}}}*/

template <class T, class A, class U, class B>
auto dot(const Polar<T,A> &v1, const Polar<U,B> &v2)/*{{{*/
	-> decltype(v1.r*v2.r*cos(v1.theta-v2.theta))
{
	return v1.r*v2.r*cos(v1.theta - v2.theta);
}/*}}}*/

template <class T, class A> 
T norm(const Polar<T,A> &v)/*{{{*/
{
	return v.r;
}/*}}}*/
template <class T, class A> 
T sqrnorm(const Polar<T,A> &v)/*{{{*/
{
	return v.r*v.r;
}/*}}}*/
template <class T, class A, class U, class B> 
auto angle(const Polar<T,A> &v1, const Polar<U,B> &v2)/*{{{*/
	-> decltype(abs(v1.theta-v2.theta))
{
	return abs(v1.theta-v2.theta);
}/*}}}*/
template <class T, class A> 
A angle(const Polar<T,A> &v1)/*{{{*/
{
	return v1.theta;
}/*}}}*/

template <class T, class A>
Polar<T,A> unit(const Polar<T,A> &p)/*{{{*/
{
	return {copysign(1,p.r),p.theta};
}/*}}}*/

template <class T, class A> 
bool is_unit(const Polar<T,A> &v)/*{{{*/
{
	return equal(abs(v.r),1);
}/*}}}*/
template <class T, class A, class U, class B> 
auto ortho(const Polar<T,A> &v, const Polar<U,B> &n)/*{{{*/
	-> decltype(v-dot(v,n)*n)
{
	assert(is_unit(n));

	return v-dot(v,n)*n;
}/*}}}*/
template <class T, class A, class U, class B> 
auto proj(const Polar<T,A> &v, const Polar<U,B> &n)/*{{{*/
	-> decltype(dot(v,n)*n)
{
	assert(is_unit(n));

	return dot(v,n)*n;
}/*}}}*/
template <class T, class A> 
Polar<T,A> rot(const Polar<T,A> &v, A theta)/*{{{*/
{
	return {v.r, v.theta + theta};
}/*}}}*/

template <class T, class A>
Polar<T,A> normalize(Polar<T,A> p)/*{{{*/
{
	if(p.r < 0)
	{
		p.r = -p.r;
		p.theta += M_PI;
	}

	p.theta = mod(p.theta, -M_PI, M_PI);

	assert(p.r >= 0);
	assert(p.theta >= A(-M_PI) && p.theta < A(M_PI));

	return p;
}/*}}}*/

}}} // namespace s3d::math::r2

// $Id: vector.hpp 2706 2010-05-27 23:18:29Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

