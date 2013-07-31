/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

namespace nih {

// maps a point in spherical coordinates to the unit sphere
inline NIH_HOST NIH_DEVICE Vector3f from_spherical_coords(const Vector2f& uv)
{
	return Vector3f(
		cosf(uv[0])*sinf(uv[1]),
		sinf(uv[0])*sinf(uv[1]),
		cosf(uv[1]) );
}
// computes the spherical coordinates of a 3d point
inline NIH_HOST NIH_DEVICE Vector2f to_spherical_coords(const Vector3f& vec)
{
    // to spherical in [0,2pi] (phi) and [0,pi] (theta).
    float theta, phi;
	if (fabsf(vec[2]) >= 1.0f - 1.0e-5f)
		phi = 0.0f;
	else {
		phi = atan2f( vec[1], vec[0] );
		phi = phi<0.0f ? phi+2.0f*float(M_PI) : phi;
	}
    theta = acosf( vec[2] );
	return Vector2f(phi,theta);
}

// seedx, seedy is point on [0,1]^2.  x, y is point on radius 1 disk
inline NIH_HOST NIH_DEVICE Vector2f square_to_unit_disk(const Vector2f seed)
{
	float phi, r;

	float a = 2*seed[0] - 1;		// (a,b) is now on [-1,1]^2
	float b = 2*seed[1] - 1;

	if (a > -b) {					// region 1 or 2
	   if (a > b) {					// region 1, also |a| > |b|
		   r = a;
		   phi = (M_PIf/4 ) * (b/a);
	   }
	   else       {					// region 2, also |b| > |a|
		   r = b;
		   phi = (M_PIf/4) * (2 - (a/b));
	   }
	}
	else {							// region 3 or 4
	   if (a < b) {					// region 3, also |a| >= |b|, a != 0
			r = -a;
			phi = (M_PIf/4) * (4 + (b/a));
	   }
	   else       {					// region 4, |b| >= |a|, but a==0 and b==0 could occur.
			r = -b;
			phi = b != 0 ? (M_PIf/4) * (6 - (a/b)) : 0;
	   }
	}

	return Vector2f(
		r * cosf(phi),
		r * sinf(phi) );
}

// diskx, disky is point on radius 1 disk.  x, y is point on [0,1]^2
inline NIH_HOST NIH_DEVICE Vector2f unit_disk_to_square(const Vector2f disk)
{
	float r = sqrtf( disk[0]*disk[0] + disk[1]*disk[1] );
	float phi = atan2f( disk[1], disk[0] );
	float a, b;
	if (phi < -M_PIf/4) phi += 2*M_PIf;	// in range [-pi/4,7pi/4]
	if ( phi < M_PIf/4) {				// region 1
		a = r;
		b = phi * a / (M_PIf/4);
	}
	else if ( phi < 3*M_PIf/4 ) {		// region 2
		b = r;
		a = -(phi - M_PIf/2) * b / (M_PIf/4);
	}
	else if ( phi < 5*M_PIf/4 ) {		// region 3
		a = -r;
		b = (phi - M_PIf) * a / (M_PIf/4);
	}
	else {								// region 4
		b = -r;
		a = -(phi - 3*M_PIf/2) * b / (M_PIf/4);
	}

	return Vector2f(
		(a + 1) / 2,
		(b + 1) / 2 );
}

// maps the unit square to the hemisphere with a cosine-weighted distribution
inline NIH_HOST NIH_DEVICE Vector3f square_to_cosine_hemisphere(const Vector2f& uv)
{
	const Vector2f disk = square_to_unit_disk( uv );
	const float r2 = disk[0]*disk[0] + disk[1]*disk[1];
	return Vector3f(
		disk[0],
		disk[1],
        sqrtf( max(1.0f - r2,0.0f) ) );
//	const float cosTheta = sqrtf(std::max(uv[1],0.0f));
//	const float sinTheta = sqrtf(std::max(1.0f - cosTheta*cosTheta,0.0f));
//	const float phi = uv[0] * 2.0f * float(M_PI);
//
//	return Vector3f(
//		cosf(phi)*sinTheta,
//		sinf(phi)*sinTheta,
//		cosTheta );
}
// inverts the square to cosine-weighted hemisphere mapping
inline NIH_HOST NIH_DEVICE Vector2f cosine_hemisphere_to_square(const Vector3f& dir)
{
	return unit_disk_to_square( Vector2f( dir[0], dir[1] ) );
}
// maps the unit square to the sphere with a uniform distribution
inline NIH_HOST NIH_DEVICE Vector3f uniform_square_to_sphere(const Vector2f& uv)
{
	const float cosTheta = uv[1]*2.0f - 1.0f;
	const float sinTheta = fast_sqrt(max(1.0f - cosTheta*cosTheta,0.0f));
	const float phi = uv[0] * 2.0f * float(M_PI);

	return Vector3f(
		fast_cos(phi)*sinTheta,
		fast_sin(phi)*sinTheta,
		cosTheta );
}
// maps the sphere to a unit square with a uniform distribution
inline NIH_HOST NIH_DEVICE Vector2f uniform_sphere_to_square(const Vector3f& vec)
{
	const float cosTheta = vec[2];
	float phi;
	if (fabsf(vec[2]) >= 1.0f - 1.0e-5f)
		phi = 0.0f;
	else {
		phi = atan2f( vec[1], vec[0] );
		phi = phi<0.0f ? phi+2.0f*float(M_PI) : phi;
	}
	return Vector2f(phi/(2.0f*float(M_PI)),(cosTheta+1.0f)*0.5f);
}

} // namespace nih
