#ifndef S3D_MATH_PLANE_H
#define S3D_MATH_PLANE_H

#include "fwd.h"
#include "vector.h"

namespace s3d { namespace math
{

template <class T, int D>
class Plane
{
public:
	Plane() {}
	Plane(const Vector<T,D> &n, T d);
	Plane(const Point<T,D> &pt, const Vector<T,D> &n);
	Plane(const Point<T,D> &pt1, const Point<T,D> &pt2, const Point<T,D> &pt3);

	Vector<T,D> n;
	T d;
};

template <class T, int D>
T intersect(const Plane<T,D> &p, const Ray<T,D> &r);

template <class T, int D>
T distance(const Point<T,D> &pt, const Plane<T,D> &plane);

template <class T, int D>
T distance(const Plane<T,D> &plane, const Point<T,D> &pt);

template <class T>
T signed_distance(const Point<T,3> &pt, const Plane<T,3> &plane);

template <class T>
T signed_distance(const Plane<T,4> &plane, const Point<T,3> &pt);

}} // namespace s3d::math

#include "plane.hpp"

#endif
