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

#ifndef S3D_MATH_AXIS_ANGLE_H
#define S3D_MATH_AXIS_ANGLE_H

#include "../coords.h"
#include "axis_angle_space.h"

namespace s3d { namespace math { namespace r3
{

template <class T, class A>
struct AxisAngle
	: coords<AxisAngle<T,A>,axis_angle_space<T,A>>
{
private:
	typedef coords<AxisAngle,axis_angle_space<T,A>> coords_base;
public:
	using coords_base::axis;
	using coords_base::angle;

	AxisAngle() {}

	template <class U, class B>
	AxisAngle(const AxisAngle<U,B> &that)
		: coords_base(that) {}

	AxisAngle(const UnitVector<T,3> &axis, A angle)
		: coords_base(axis, angle) {}

	bool operator==(const AxisAngle &that) const;

	AxisAngle operator-() const;
	AxisAngle &operator+=(const AxisAngle &that);
	AxisAngle &operator-=(const AxisAngle &that);
};

template <class T, class A>
std::ostream &operator<<(std::ostream &out, const AxisAngle<T,A> &aa);

template <class T, class A>
A angle(const AxisAngle<T,A> &aa);

template <class T, class A>
UnitVector<T,3> axis(const AxisAngle<T,A> &aa);

template <class T, class A>
AxisAngle<T,A> normalize(AxisAngle<T,A> aa);

template <class T, class A>
AxisAngle<T,A> inv(const AxisAngle<T,A> &aa);

template <class T, class A>
AxisAngle<T,A> &inv_inplace(AxisAngle<T,A> &aa);

}}} // namespace s3d::math::r3

#include "axis_angle.hpp"

#endif
