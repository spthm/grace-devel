#include <limits>
#include "../math/real.h"
#include "face.h"
#include "../math/plane.h"

namespace s3d { namespace math
{

template <class T, class F> 
T intersect(const triangle_view<F> &tri, const Ray<T,3> &r)
{
	Plane<T,3> plane(tri[0], tri[1], tri[2]);

	auto t = intersect(plane, r);
	if(t < 0)
		return -std::numeric_limits<T>::max();

	auto q0 = r.point_at(t) - tri[0],
		 q1 = tri[1] - tri[0],
		 q2 = tri[2] - tri[0];

	ortho_plane oplane = (ortho_plane)max_index(plane.n.x,plane.n.y,plane.n.z);

	auto p0 = proj(q0, oplane),
		 p1 = proj(q1, oplane),
		 p2 = proj(q2, oplane);

	T den = p1.x*p2.y - p2.x*p1.y,
	    a = (p0.x*p2.y - p2.x*p1.y)/den,
	    b = (p1.x*p0.y - p0.x*p1.y)/den;

	if(a>=0 && b>=0 && (a+b) <= 1)
		return t;
	return -REAL_MAX;
}

}} // namespace s3d::math
