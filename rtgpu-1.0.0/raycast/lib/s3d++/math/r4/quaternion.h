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

#ifndef S3D_MATH_R4_QUATERNION_H
#define S3D_MATH_R4_QUATERNION_H

#include "../r3/vector.h"
#include "../real.h"
#include "../../util/concepts.h"
#include "../coords.h"
#include "quaternion_space.h"
#include "fwd.h"

namespace s3d { namespace math { namespace r4
{

template <class T>
struct is_quaternion
{
	static const bool value = false;
};

template <class T>
struct is_quaternion<T&> : is_quaternion<T> {};


template <class T>
struct Quaternion
	: coords<Quaternion<T>,quaternion_space<T>>
{
private:
	typedef coords<Quaternion,quaternion_space<T>> coords_base;
public:
	using coords_base::w;
	using coords_base::x;
	using coords_base::y;
	using coords_base::z;

	template <class U> struct rebind { typedef Quaternion<U> type; };

	Quaternion() {}
	Quaternion(T w, T x=0, T y=0, T z=0) : coords_base(w,x,y,z) {}
	Quaternion(const Vector<T,3> &v) : coords_base(0,v.x,v.y,v.z) {}

	template <class Q, class 
		= typename std::enable_if<is_quaternion<Q>::value>::type>
	Quaternion(const Q &that) : coords_base(that) {}

	template <class Q,
			 class = typename std::enable_if<is_quaternion<Q>::value>::type>
	auto operator +=(const Q &that) -> Quaternion &;

	template <class Q, 
			 class = typename std::enable_if<is_quaternion<Q>::value>::type>
	auto operator -=(const Q &that) -> Quaternion &;

	template <class Q, 
			 class = typename std::enable_if<is_quaternion<Q>::value>::type>
	auto operator *=(const Q &that) -> Quaternion &;

	template <class Q, 
			 class = typename std::enable_if<is_quaternion<Q>::value>::type>
	auto operator /=(const Q &that) -> Quaternion &;

	Quaternion &operator +=(T d);
	Quaternion &operator -=(T d);
	Quaternion &operator /=(T d);
	Quaternion &operator *=(T d);
};

template <class T>
struct is_quaternion<Quaternion<T>>
{
	static const bool value = true;
};

template <class T>
struct quaternion_like : concept<T> {};

} // namespace r4

using r4::Quaternion;
using r4::quaternion_like;
using r4::is_quaternion;

}

template <class T>
struct concept_arg<T, typename std::enable_if<math::r4::is_quaternion<T>::value>::type>
{
	typedef math::r4::quaternion_like<T> type;
};

} // namespace s3d::math::r4

#include "quaternion.hpp"

#endif


// $Id: quaternion.h 3095 2010-09-01 20:57:28Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

