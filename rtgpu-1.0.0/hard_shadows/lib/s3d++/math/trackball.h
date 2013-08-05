#ifndef S3D_MATH_ARCBALL_H
#define S3D_MATH_ARCBALL_H

#include "fwd.h"

namespace s3d { namespace math
{

enum trackball_method
{
	SHOEMAKE,
	BELL
};

r4::unit_quaternion trackball(const r2::point &p1, const r2::point &p2,
							  const r2::point &center, real radius,
							  trackball_method method=BELL);

}}

#endif
