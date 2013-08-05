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

#ifndef  S3D_MATH_R2_POLAR_H
#define  S3D_MATH_R2_POLAR_H

#include <boost/operators.hpp>
#include "../coords.h"
#include "polar_space.h"
#include "fwd.h"

namespace s3d { namespace math { namespace r2
{

template <class T, class A>
struct Polar
	: coords<Polar<T,A>,polar_space<T,A>>
{
private:
	typedef coords<Polar,polar_space<T,A>> coords_base;
public:
	using coords_base::r;
	using coords_base::theta;

	template <class U> struct rebind { typedef Polar<U,A> type; };

	Polar() : coords_base() {}
	Polar(const std::tuple<T,real> &a) : coords_base(a) {}
	explicit Polar(const Vector<T,2> &v);

	template <class U, class B> Polar(const Polar<U,B> &v) : coords_base(v) {}
	Polar(const Polar &v) = default;

	Polar(T r, A theta) : coords_base(r,theta) {}

	Polar operator -() const;
	bool operator==(const Polar &that) const;

#if GCC_VERSION >= 40500
	explicit operator Vector<T,2>() const;
#else
	operator Vector<T,2>() const;
#endif

	template <class U, class A2>
	Polar &operator +=(const Polar<U,A2> &that);
	template <class U, class A2>
	Polar &operator -=(const Polar<U,A2> &that);

	template <class U, class A2>
	Polar &operator *=(const Polar<U,A2> &that);
	template <class U, class A2>
	Polar &operator /=(const Polar<U,A2> &that);

//	Polar &operator +=(T d);
//	Polar &operator -=(T d);
	Polar &operator /=(T d);
	Polar &operator *=(T d);
};

}} // namespace math::r2

template <class T, class A>
struct order<math::r2::Polar<T,A>>
{
	static const int value = order<T>::value + 1;
};

} // namespace s3d

#include "polar.hpp"

#endif
