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
#include "point.h"

namespace s3d { 

namespace math 
{ 

namespace r3
{

template <class T, class A>
Spherical<T,A>::Spherical(const Vector<T,3> &cart)/*{{{*/
{
	r = norm(cart);
	theta = atan2(cart.y,cart.x);
	phi = acos(cart.z/r);
}/*}}}*/

template <class T, class A>
Spherical<T,A>::Spherical(const Point<T,3> &cart)/*{{{*/
{
	auto v = cart - math::origin<T,3>();

	r = norm(v);
	theta = atan2(v.y,v.x);
	phi = acos(v.z/r);
}/*}}}*/

template <class T, class A>
Spherical<T,A>::operator Vector<T,3>() const/*{{{*/
{
	return { r*cos(theta)*sin(phi), r*sin(theta)*sin(phi), r*cos(phi) };
}/*}}}*/

template <class T, class A> 
bool Spherical<T,A>::operator==(const Spherical &that) const/*{{{*/
{
	auto p1 = normalize(*this),
		 p2 = normalize(that);

	if(!equal(p1.r,p2.r))
		return false;
	else if(equal(p1.r,0))
		return true;

	return equal(p1.theta, p2.theta) && equal(p1.phi, p2.phi);
}/*}}}*/

template <class T, class A> 
Spherical<T,A> Spherical<T,A>::operator -() const/*{{{*/
{
	return {-r, theta, phi};
}/*}}}*/

template <class T, class A>
Spherical<T,A> &Spherical<T,A>::operator +=(const Spherical &v)/*{{{*/
{
	return *this = Spherical(Vector<T,3>(*this) + Vector<T,3>(v));
}/*}}}*/

template <class T, class A>
Spherical<T,A> &Spherical<T,A>::operator -=(const Spherical &v)/*{{{*/
{
	return *this = Spherical(Vector<T,3>(*this) - Vector<T,3>(v));
}/*}}}*/

template <class T, class A> 
Spherical<T,A> &Spherical<T,A>::operator *=(T d)/*{{{*/
{
	r *= d;
	return *this;
}/*}}}*/
template <class T, class A> 
Spherical<T,A> &Spherical<T,A>::operator /=(T d)/*{{{*/
{
	r /= d;
	return *this;
}/*}}}*/

template <class T, class A> 
std::ostream &operator<<(std::ostream &out, const Spherical<T,A> &v)/*{{{*/
{
	return out << '[' << (equal(v.r,0)?0:v.r)
			   << '<' << (equal(v.theta,0)?0:deg(v.theta)) << "°"
			   << '<' << (equal(v.phi,0)?0:deg(v.phi)) << "°"
			   << ']';
}/*}}}*/
template <class T, class A> 
std::istream &operator>>(std::istream &in, Spherical<T,A> &v)/*{{{*/
{
	if(in.peek() == '[')
		in.ignore();

	Spherical<T,A> aux;

	in >> aux.r;
	if(in.peek() == '<')
		in.ignore();

	in >> aux.theta;

	if(!in)
		return in;

	if(in.peek() == 'o' || in.peek() == "°"[0])
	{
		aux.theta = rad(aux.theta);
		in.ignore();
	}

	if(in.peek() == '<')
		in.ignore();

	in >> aux.phi;

	if(!in)
		return in;

	if(in.peek() == 'o' || in.peek() == "°"[0])
	{
		aux.phi = rad(aux.theta);
		in.ignore();
	}


	if(!in.eof() && in.peek() == ']')
		in.ignore();

	if(in)
		v = aux;

	return in;
}/*}}}*/

template <class T, class A, class U, class B>
auto dot(const Spherical<T,A> &v1, const Spherical<U,B> &v2)/*{{{*/
	-> decltype(v1.r*v2.r*cos(v1.theta+v2.theta))
{
	return v1.r*v2.r*(cos(v2.theta-v1.theta + v2.phi-v1.phi) +	
					  cos(v2.theta-v1.theta - v2.phi+v1.phi) -
					  cos(v2.theta-v1.theta + v2.phi+v1.phi) -
					  cos(v2.theta-v1.theta - v2.phi-v1.phi) +
					  2*cos(v2.phi + v1.phi) +
					  2*cos(v2.phi - v1.phi))/4;
}/*}}}*/

template <class T, class A> 
T norm(const Spherical<T,A> &v)/*{{{*/
{
	return v.r;
}/*}}}*/
template <class T, class A> 
T sqrnorm(const Spherical<T,A> &v)/*{{{*/
{
	return v.r*v.r;
}/*}}}*/
template <class T, class A, class U, class B> 
auto angle(const Spherical<T,A> &v1, const Spherical<U,B> &v2)/*{{{*/
	-> decltype(mod(abs(v1.theta-v2.theta),-M_PI,M_PI))
{
	return mod(abs(v1.theta-v2.theta),-M_PI,M_PI);
}/*}}}*/

template <class T, class A>
Spherical<T,A> unit(const Spherical<T,A> &p)/*{{{*/
{
	return { copysign(1,p.r), p.theta, p.phi };
}/*}}}*/

template <class T, class A> 
bool is_unit(const Spherical<T,A> &v)/*{{{*/
{
	return equal(abs(v.r),1);
}/*}}}*/
template <class T, class A, class U, class B> 
auto ortho(const Spherical<T,A> &v, const Spherical<U,B> &n)/*{{{*/
	-> decltype(v-dot(v,n)*n)
{
	assert(is_unit(n));
	return v-dot(v,n)*n;
}/*}}}*/
template <class T, class A, class U, class B> 
auto proj(const Spherical<T,A> &v, const Spherical<U,B> &n)/*{{{*/
	-> decltype(dot(v,n)*n)
{
	assert(is_unit(n));

	return dot(v,n)*n;
}/*}}}*/
template <class T, class A> 
Spherical<T,A> rot(const Spherical<T,A> &v, A theta, A phi)/*{{{*/
{
	return { v.r, v.theta + theta, v.phi+phi };
}/*}}}*/

template <class T, class A>
Spherical<T,A> normalize(Spherical<T,A> p)/*{{{*/
{
	p.phi = mod(p.phi,0,2*M_PI);

	if(p.phi > A(M_PI))
	{
		p.theta += M_PI;
		p.phi = 2*M_PI-p.phi;
	}

	if(p.r < 0)
	{
		p.r = -p.r;
		p.theta += M_PI;
		p.phi = M_PI-p.phi;
	}

	p.theta = mod(p.theta, -M_PI, M_PI);

	assert(p.r >= 0);
	assert(p.theta >= A(-M_PI) && p.theta < A(M_PI));
	assert(p.phi >= 0 && p.theta < A(M_PI));

	return p;
}/*}}}*/

}}} // namespace s3d::math::r3

// $Id: vector.hpp 2706 2010-05-27 23:18:29Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

