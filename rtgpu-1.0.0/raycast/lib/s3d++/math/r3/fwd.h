/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License version 3 as 
	published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public License
	along with S3D++.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef S3D_MATH_R3_FWD_H
#define S3D_MATH_R3_FWD_H

#include <list>
#include "../fwd.h"

namespace s3d { namespace math { namespace r3 {

typedef Box<real,3> box;
typedef Box<double,3> dbox;
typedef Box<size_t,3> ubox;
typedef Box<int,3> ibox; 
typedef Box<short,3> sbox;
typedef Box<float,3> fbox;

typedef Size<real,3> size;
typedef Size<double,3> dsize;
typedef Size<int,3> isize; 
typedef Size<short,3> ssize;
typedef Size<float,3> fsize;

typedef Point<real,3> point;
typedef Point<double,3> dpoint;
typedef Point<int,3> ipoint; 
typedef Point<short,3> spoint;
typedef Point<float,3> fpoint;

typedef Vector<real,3> vector;
typedef Vector<double,3> dvector;
typedef Vector<int,3> ivector; 
typedef Vector<short,3> svector;
typedef Vector<float,3> fvector;

typedef UnitVector<real,3> unit_vector;
typedef UnitVector<double,3> dunit_vector;
typedef UnitVector<float,3> funit_vector;

template <class T, class A=T> struct Spherical;
typedef Spherical<real> spherical;
typedef Spherical<double> dspherical;
typedef Spherical<int,real> ispherical; 
typedef Spherical<short,real> sspherical;
typedef Spherical<float> fspherical;

typedef Matrix<real,3,3> matrix;
typedef Matrix<double,3,3> dmatrix;
typedef Matrix<int,3,3> imatrix; 
typedef Matrix<short,3,3> smatrix;
typedef Matrix<float,3,3> fmatrix;

typedef Transform<real,3> transform;
typedef LinearTransform<real,3> linear_transform;
typedef AffineTransform<real,3> affine_transform;
typedef ProjTransform<real,3> proj_transform;

typedef ParamCoord<real,3> param_coord;
typedef ParamCoord<double,3> dparam_coord;
typedef ParamCoord<float,3> fparam_coord;

typedef Ray<real,3> ray;

typedef Cone<real,3> cone;
typedef Plane<real,3> plane;

inline namespace rotation_frames
{
enum rotation_frame /*{{{*/
{
	FRAME_DYNAMICS = 1<<0,
	FRAME_STATIC   = 0<<0,
	FRAME_ROTATING = 1<<0,

	REPETITION	   = 1<<1,
	REP_NO		   = 0<<1,
	REP_YES		   = 1<<1,

	PARITY		   = 1<<2,
	PAR_EVEN       = 0<<2,
	PAR_ODD        = 1<<2,

	AXIS		   = 3<<3,
	AXIS_X		   = 0<<3,
	AXIS_Y		   = 1<<3,
	AXIS_Z		   = 2<<3,

	XYZs = FRAME_STATIC | AXIS_X | PAR_EVEN | REP_NO,
	XYXs = FRAME_STATIC | AXIS_X | PAR_EVEN | REP_YES,
	XZYs = FRAME_STATIC | AXIS_X | PAR_ODD  | REP_NO,
	XZXs = FRAME_STATIC | AXIS_X | PAR_ODD  | REP_YES,
	YZXs = FRAME_STATIC | AXIS_Y | PAR_EVEN | REP_NO,
	YZYs = FRAME_STATIC | AXIS_Y | PAR_EVEN | REP_YES,
	YXZs = FRAME_STATIC | AXIS_Y | PAR_ODD  | REP_NO,
	YXYs = FRAME_STATIC | AXIS_Y | PAR_ODD  | REP_YES,
	ZXYs = FRAME_STATIC | AXIS_Z | PAR_EVEN | REP_NO,
	ZXZs = FRAME_STATIC | AXIS_Z | PAR_EVEN | REP_YES,
	ZYXs = FRAME_STATIC | AXIS_Z | PAR_ODD  | REP_NO,
	ZYZs = FRAME_STATIC | AXIS_Z | PAR_ODD  | REP_YES,

	ZYXr = FRAME_ROTATING | AXIS_X | PAR_EVEN | REP_NO,
	XYXr = FRAME_ROTATING | AXIS_X | PAR_EVEN | REP_YES,
	YZXr = FRAME_ROTATING | AXIS_X | PAR_ODD  | REP_NO,
	XZXr = FRAME_ROTATING | AXIS_X | PAR_ODD  | REP_YES,
	XZYr = FRAME_ROTATING | AXIS_Y | PAR_EVEN | REP_NO,
	YZYr = FRAME_ROTATING | AXIS_Y | PAR_EVEN | REP_YES,
	ZXYr = FRAME_ROTATING | AXIS_Y | PAR_ODD  | REP_NO,
	YXYr = FRAME_ROTATING | AXIS_Y | PAR_ODD  | REP_YES,
	YXZr = FRAME_ROTATING | AXIS_Z | PAR_EVEN | REP_NO,
	ZXZr = FRAME_ROTATING | AXIS_Z | PAR_EVEN | REP_YES,
	XYZr = FRAME_ROTATING | AXIS_Z | PAR_ODD  | REP_NO,
	ZYZr = FRAME_ROTATING | AXIS_Z | PAR_ODD  | REP_YES,

	// ISO 31-11
	// [x y z] = r[cos(theta)sin(phi) sin(theta)*sin(phi) cos(phi)]
	// +Z points up, +X points forward, +Y points left
	MATH_FRAME = ZYXr 
};/*}}}*/
}

template <rotation_frame F,class T=real> struct Euler;

template <class T=real, class A=T> struct AxisAngle;
typedef AxisAngle<> axis_angle;
typedef AxisAngle<double> daxis_angle;
typedef AxisAngle<int,real> iaxis_angle; 
typedef AxisAngle<short,real> saxis_angle;
typedef AxisAngle<float> faxis_angle;

template <class T> struct Frustum;
typedef Frustum<real> frustum;
typedef Frustum<double> dfrustum;

} // namespace r3

using r3::Spherical;
using r3::spherical;
using r3::dspherical;
using r3::ispherical;
using r3::sspherical;
using r3::fspherical;

using r3::Euler;
using namespace r3::rotation_frames;

using r3::AxisAngle;
using r3::axis_angle;
using r3::daxis_angle;
using r3::iaxis_angle;
using r3::saxis_angle;
using r3::faxis_angle;

} // namespace math

namespace r3 = math::r3;

} 

#endif

// $Id: fwd.h 3143 2010-09-21 18:36:05Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

