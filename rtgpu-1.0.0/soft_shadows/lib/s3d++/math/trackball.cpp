#include "rotation.h"
#include "trackball.h"
#include <iostream>

namespace s3d { namespace math
{

// K. Shoemake, "Arcball Rotation Control", Graphics Gems, P.S. Heckbert, ed.,
// vol IV, pp.175-192, Academic Press, 1994

namespace detail
{
	r3::vector project_to_sphere(const r2::point &p, 
							     const r2::point &c, real r,
								 trackball_method method)
	{
		r2::vector v = (p-c)/r;

		real d2 = sqrnorm(v);

		switch(method)
		{
		case SHOEMAKE:
			if(d2 <= 1) // inside sphere?
				return {v.x, v.y, sqrt(1-d2)};
			else
				return {v.x, v.y, 0};

		default:
			assert(false);
			// fall-through
		case BELL:
			if(d2 <= 0.5) // inside sphere?
				return {v.x, v.y, sqrt(1-d2)};
			else // else, treat like it's on a hyperbola
				return {v.x, v.y, 0.5/sqrt(d2)};
		}
	}
}

r4::unit_quaternion trackball(const r2::point &p1, const r2::point &p2, 
							const r2::point &c, real r,
							trackball_method method)
{
	// no rotation?
	if(p1 == p2)
		return {};

	auto pA = unit(detail::project_to_sphere(p1,c,r, method)),
		 pB = unit(detail::project_to_sphere(p2,c,r, method));

	return to_quaternion(unit(cross(pA,pB)), acos(dot(pA,pB)));
}

}}
