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

#ifndef S3D_MATH_R4_FWD_H
#define S3D_MATH_R4_FWD_H

#include "../fwd.h"

namespace s3d { namespace math { namespace r4 {

typedef Box<real,4> box;
typedef Box<double,4> dbox;
typedef Box<size_t,4> ubox;
typedef Box<int,4> ibox; 
typedef Box<short,4> sbox;
typedef Box<float,4> fbox;

typedef Size<real,4> size;
typedef Size<double,4> dsize;
typedef Size<int,4> isize; 
typedef Size<short,4> ssize;
typedef Size<float,4> fsize;

typedef Point<real,4> point;
typedef Point<double,4> dpoint;
typedef Point<int,4> ipoint; 
typedef Point<short,4> spoint;
typedef Point<float,4> fpoint;

typedef Vector<real,4> vector;
typedef Vector<double,4> dvector;
typedef Vector<int,4> ivector; 
typedef Vector<short,4> svector;
typedef Vector<float,4> fvector;

typedef UnitVector<real,4> unit_vector;
typedef UnitVector<double,4> dunit_vector;
typedef UnitVector<float,4> funit_vector;

typedef Matrix<real,4,4> matrix;
typedef Matrix<double,4,4> dmatrix;
typedef Matrix<int,4,4> imatrix; 
typedef Matrix<short,4,4> smatrix;
typedef Matrix<float,4,4> fmatrix;

template <class T> struct Quaternion;
typedef Quaternion<real> quaternion;
typedef Quaternion<double> dquaternion;
typedef Quaternion<float> fquaternion;
typedef Quaternion<int> iquaternion;

template <class T> struct UnitQuaternion;
typedef UnitQuaternion<real> unit_quaternion;
typedef UnitQuaternion<double> dunit_quaternion;
typedef UnitQuaternion<float> funit_quaternion;

} // namespace r4

using r4::Quaternion;
using r4::quaternion;
using r4::dquaternion;
using r4::fquaternion;
using r4::iquaternion;

using r4::UnitQuaternion;
using r4::unit_quaternion;
using r4::dunit_quaternion;
using r4::funit_quaternion;

}

namespace r4 = math::r4;

} // namespace s3d::math

#endif

// $Id: fwd.h 2985 2010-08-19 21:07:58Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

