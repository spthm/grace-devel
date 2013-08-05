/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License 
	version 3 as published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public 
	License along with S3D++. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef S3D_MATH_R3_ROTATION_H
#define S3D_MATH_R3_ROTATION_H

#include "fwd.h"
#include "../r4/fwd.h"
#include "euler.h" // por causa dos rotframe_traits
#include "../vector.h"
#include "../point.h"

namespace s3d { namespace math { namespace r3
{

// Conversion to quaternion representation

template <rotation_frame F, class T>
UnitQuaternion<T> to_quaternion(const Euler<F,T> &rot);

template <class T>
UnitQuaternion<T> to_quaternion(const Matrix<T,3,3> &m);

template <class T, class A> 
UnitQuaternion<T> to_quaternion(const AxisAngle<T,A> &aa);

template <class T, class A> 
UnitQuaternion<T> to_quaternion(const UnitVector<T,3> &axis, A angle);

template <class T> 
UnitQuaternion<T> to_quaternion(const UnitVector<T,3> &from_dir, 
							    const UnitVector<T,3> &from_up, 
							    const UnitVector<T,3> &to_dir, 
							    const UnitVector<T,3> &to_up);
// Conversion to euler representation

template <rotation_frame F, class T>
Euler<F,T> to_euler(const Matrix<T,3,3> &m);

template <rotation_frame F, class T, class A> 
Euler<F,T> to_euler(const AxisAngle<T,A> &aa);

template <rotation_frame F, class T>
Euler<rotframe_traits<F>::revert, T> to_rev_euler(const Matrix<T,3,3>&m);

template <rotation_frame F, class T, class A> 
Euler<rotframe_traits<F>::revert,T> to_rev_euler(const AxisAngle<T,A> &aa);

// Conversion to matrix representation

template <rotation_frame F, class T>
Matrix<T,3,3> to_rot_matrix(const Euler<F,T> &rot);

template <class T, class A>
Matrix<T,3,3> to_rot_matrix(const AxisAngle<T,A> &a);

// Rotação de vetores

template <class T, class V, class A>
auto rotx(const V &v, A angle)
	-> typename std::enable_if<(is_vector<V>::value && is_point<V>::value)
									&& V::dim==3, V>::type;

template <class T, class V, class A>
auto roty(const V &v, A angle)
	-> typename std::enable_if<(is_vector<V>::value && is_point<V>::value)
									&& V::dim==3, V>::type;

template <class T, class V, class A>
auto rotz(const V &v, A angle)
	-> typename std::enable_if<(is_vector<V>::value && is_point<V>::value)
									&& V::dim==3, V>::type;

template <class T, class V, class A> 
auto rot(const V &v, const AxisAngle<T,A> &aa)
	-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value)
									&& V::dim==3, V>::type;

template <class V>
V conv_rotframe(const V &v, rotation_frame from, rotation_frame to);

} // namespace r3

namespace r4
{
	template <class A=real, class T> 
	A angle(const UnitQuaternion<T> &q);

	template <class T> 
	UnitVector<T,3> axis(const UnitQuaternion<T> &q);

	template <rotation_frame F, class T> 
	auto to_rev_euler(const UnitQuaternion<T>&q)
		-> r3::Euler<r3::rotframe_traits<F>::revert,T>;

	template <rotation_frame F, class T> 
	r3::Euler<F,T> to_euler(const UnitQuaternion<T> &q);

	template <class T>
	Matrix<T,3,3> to_rot_matrix(const UnitQuaternion<T> &q);

	template <class T, class V> 
	auto rot(const V &v, const UnitQuaternion<T> &q)
		-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value)
										&& V::dim==3, V>::type;
}

namespace r3
{
	using r4::axis;
	using r4::angle;
	using r4::to_euler;
	using r4::to_rev_euler;
	using r4::to_rot_matrix;
	using r4::rot;
}

}} // namespace s3d::math

#include "rotation.hpp"

#endif

// $Id: rotation.h 3143 2010-09-21 18:36:05Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

