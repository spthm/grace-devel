/*
	Copyright (c) 2010, Rodolfo Schulz de Lima.

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

// vim: nocp:ci:sts=4:fdm=marker:fmr={{{,}}}
// vi: ai:sw=4:ts=8

#ifndef  S3D_MATH_R3_SPHERICAL_H
#define  S3D_MATH_R3_SPHERICAL_H

#include <boost/operators.hpp>
#include "../coords.h"
#include "spherical_space.h"
#include "fwd.h"

namespace s3d { namespace math { namespace r3
{

template <class T, class A>
struct Spherical
	: coords<Spherical<T,A>,spherical_space<T,A>>
{
private:
	typedef coords<Spherical,spherical_space<T,A>> coords_base;
public:
	using coords_base::r;
	using coords_base::theta;
	using coords_base::phi;

	template <class U> struct rebind { typedef Spherical<U,A> type; };

	Spherical() : coords_base() {}
	Spherical(const std::tuple<T,A,A> &a) : coords_base(a) {}
	explicit Spherical(const Vector<T,3> &v);
	explicit Spherical(const Point<T,3> &v);

	template <class U, class B> 
	Spherical(const Spherical<U,B> &v) : coords_base(v) {}
	Spherical(const Spherical &v) = default;

	Spherical(T r, A theta, A phi) : coords_base(r,theta,phi) {}

#if GCC_VERSION >= 40500
	explicit operator Vector<T,3>() const;
#else
	operator Vector<T,3>() const;
#endif

	Spherical operator -() const;
	bool operator==(const Spherical &that) const;

	Spherical &operator +=(const Spherical &that);
	Spherical &operator -=(const Spherical &that);

//	Spherical &operator *=(const Spherical &that);
//	Spherical &operator /=(const Spherical &that);

//	Spherical &operator +=(T d);
//	Spherical &operator -=(T d);
	Spherical &operator /=(T d);
	Spherical &operator *=(T d);
};

}} // namespace math::r3

template <class T, class A>
struct order<math::r3::Spherical<T,A>>
{
	static const int value = order<T>::value + 1;
};

} // namespace s3d

#include "spherical.hpp"

#endif
