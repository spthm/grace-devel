#include "point.h"

namespace s3d { namespace math
{

template <class T, int D>
Plane<T,D>::Plane(const Vector<T,D> &_n, T _d)
	: n(_n), d(_d)
{
}

template <class T, int D>
Plane<T,D>::Plane(const Point<T,D> &pt, const Vector<T,D> &_n)
	: n(_n), d(-dot(_n, pt-origin<T,D>()))
{
}

template <class T, int D>
Plane<T,D>::Plane(const Point<T,D> &pt1, const Point<T,D> &pt2, 
				  const Point<T,D> &pt3)
	: n(cross(pt2-pt1, pt3-pt1))
{
	d = -dot(n, pt1-origin<T,D>());
}

template <class T, int D>
T intersect(const Plane<T,D> &p, const Ray<T,D> &r)
{
	T den = dot(p.n, r.dir);
	if(equal(den,0))
		return -std::numeric_limits<T>::max();
	else
		return -(dot(p.n, r.origin-origin<T,D>()) + p.d)/den;
}

template <class T, int D>
T distance(const Point<T,D> &pt, const Plane<T,D> &plane)
{
    // ref: Schneider, Eberly: Geometric Tools for Computer Graphics, pg 374
    return abs(dot(plane.n, pt-plane.p));
}

template <class T, int D>
T distance(const Plane<T,D> &plane, const Point<T,D> &pt)
{
	return distance(pt, plane);
}

template <class T>
T signed_distance(const Point<T,3> &pt, const Plane<T,3> &plane)
{
    // ref: Schneider, Eberly: Geometric Tools for Computer Graphics, pg 376
        return dot(plane.n,pt-origin<T,3>())+plane.d;
	//return plane.n*(pt - plane.p);
#if 0
    return plane.n.x*(pt.x-plane.p.x) +
		   plane.n.y*(pt.y-plane.p.y) +
		   plane.n.z*(pt.z-plane.p.z);
#endif
}

template <class T>
T signed_distance(const Plane<T,3> &plane, const Point<T,3> &pt)
{
    return signed_distance(pt, plane);
}


}} // namespace s3d::math

