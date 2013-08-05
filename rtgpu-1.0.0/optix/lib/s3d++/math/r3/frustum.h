#ifndef S3D_MATH_R3_FRUSTUM_H
#define S3D_MATH_R3_FRUSTUM_H

#include "fwd.h"
#include "unit_vector.h"
#include "point.h"

namespace s3d { namespace math { namespace r3
{

template <class T>
struct Frustum
{
	Point<T,3> pos;
	UnitVector<T,3> axis, up;

	T fov,  // horizontal fov
	  aspect, // width / height
	  near, far;

	bool operator==(const Frustum &f) const
	{
		return fov==f.fov && axis==f.axis && up==f.up &&
			   aspect == f.aspect && near==f.near && far==f.far;
	}
	bool operator!=(const Frustum &f) const
	{
		return !operator==(f);
	}
};

template <class T>
Plane<T,3> top_plane(const Frustum<T> &f);

template <class T>
Plane<T,3> bottom_plane(const Frustum<T> &f);

template <class T>
Plane<T,3> near_plane(const Frustum<T> &f);

template <class T>
Plane<T,3> far_plane(const Frustum<T> &f);

template <class T>
Plane<T,3> left_plane(const Frustum<T> &f);

template <class T>
Plane<T,3> right_plane(const Frustum<T> &f);

template <class T>
T fov_horiz(const Frustum<T> &f);

template <class T>
T fov_vert(const Frustum<T> &f);

template <class T, class R>
Frustum<T> &rot_inplace(Frustum<T> &f, const R &r);

template <class T, class R>
Frustum<T> rot(const Frustum<T> &f, const R &r);

template <class T, class R>
Frustum<T> &&rot(Frustum<T> &&f, const R &r);

}}} // namespace s3d::math::r3

#include "frustum.hpp"

#endif
