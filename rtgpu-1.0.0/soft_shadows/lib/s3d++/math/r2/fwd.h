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

#ifndef S3D_MATH_R2_FWD_H
#define S3D_MATH_R2_FWD_H

#include <list>
#include "../fwd.h"

#ifdef Complex
#	undef Complex
#endif

namespace s3d { namespace math { namespace r2 {

typedef Box<real,2> box;
typedef Box<double,2> dbox;
typedef Box<int,2> ibox; 
typedef Box<size_t,2> ubox; 
typedef Box<short,2> sbox;
typedef Box<float,2> fbox;

typedef Size<real,2> size;
typedef Size<double,2> dsize;
typedef Size<int,2> isize; 
typedef Size<short,2> ssize;
typedef Size<float,2> fsize;
typedef Size<std::size_t,2> usize;

typedef Point<real,2> point;
typedef Point<double,2> dpoint;
typedef Point<int,2> ipoint; 
typedef Point<size_t,2> upoint; 
typedef Point<short,2> spoint;
typedef Point<float,2> fpoint;

typedef Vector<real,2> vector;
typedef Vector<double,2> dvector;
typedef Vector<int,2> ivector; 
typedef Vector<short,2> svector;
typedef Vector<float,2> fvector;

typedef UnitVector<real,2> unit_vector;
typedef UnitVector<double,2> dunit_vector;
typedef UnitVector<float,2> funit_vector;

template <class T, class A=T> struct Polar;
typedef Polar<real> polar;
typedef Polar<double> dpolar;
typedef Polar<int,real> ipolar; 
typedef Polar<short,real> spolar;
typedef Polar<float> fpolar;

typedef Matrix<real,2,2> matrix;
typedef Matrix<double,2,2> dmatrix;
typedef Matrix<int,2,2> imatrix; 
typedef Matrix<short,2,2> smatrix;
typedef Matrix<float,2,2> fmatrix;

template <class T> struct Complex;
typedef Complex<real> complex;
typedef Complex<double> dcomplex;
typedef Complex<float> fcomplex;
typedef Complex<int> icomplex;

template <class T> struct UnitComplex;
typedef UnitComplex<real> unit_complex;
typedef UnitComplex<double> dunit_complex;
typedef UnitComplex<float> funit_complex;

typedef Transform<real,2> transform;
typedef AffineTransform<real,2> affine_transform;
typedef LinearTransform<real,2> linear_transform;

typedef ParamCoord<real,2> param_coord;

typedef Ray<real,2> ray;

}

using r2::Polar;
using r2::polar;
using r2::dpolar;
using r2::ipolar;
using r2::spolar;
using r2::fpolar;

using r2::Complex;
using r2::complex;
using r2::dcomplex;
using r2::fcomplex;
using r2::icomplex;

using r2::UnitComplex;
using r2::unit_complex;
using r2::dunit_complex;
using r2::funit_complex;

}

namespace r2 = math::r2;

} // namespace s3d::math::r2

#endif

// $Id: fwd.h 3130 2010-09-14 02:02:24Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

