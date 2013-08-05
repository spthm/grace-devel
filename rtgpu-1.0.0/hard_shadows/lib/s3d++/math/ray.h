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

#ifndef S3D_MATH_RAY_H
#define S3D_MATH_RAY_H

#include "vector.h"
#include "point.h"
#include "matrix.h"
#include <boost/operators.hpp>

namespace s3d { namespace math
{

template <class T, int D>
struct Ray
	: boost::multipliable<Ray<T,D>, Matrix<T,D+1,D+1>, 
	  boost::additive<Ray<T,D>, Vector<T,D>>>
{
	Ray() {}
	template <class U, class V> Ray(const Point<U,D> &o, const Vector<V,D> &d)
		: origin(o), dir(d) {}

	Point<T,D> origin;
	Vector<T,D> dir;

	Point<T,D> point_at(real t) const;

	Ray<T,D> &operator *=(const Matrix<T,D+1,D+1> &m);

	Ray<T,D> &operator +=(const Vector<T,D> &v);
	Ray<T,D> &operator -=(const Vector<T,D> &v);

	bool operator==(const Ray &that) const
		{ return origin==that.origin && dir==that.dir; }

	bool operator!=(const Ray &that) const
		{ return !operator==(that); }
};

const real RAY_EPSILON = 1e-5;

template <class T, int D> 
Ray<T,D> unit(const Ray<T,D> &r);

}} // namespace s3d::math

#include "ray.hpp"

#endif

// $Id: ray.h 2972 2010-08-18 05:22:05Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

