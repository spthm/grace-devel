/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License 
	version 3 as published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public 
	License along with S3D++. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef S3D_MATH_R2_ROTATION_H
#define S3D_MATH_R2_ROTATION_H

#include "unit_complex.h"

namespace s3d { namespace math { namespace r2
{

// Conversion to complex

template <class T=real, class A>
UnitComplex<T> to_complex(const A &ang);

template <class T>
UnitComplex<T> to_complex(const Matrix<T,2,2> &m);

// Conversion to matrix

template <class T> 
Matrix<T,2,2> to_rot_matrix(const UnitComplex<T> &c);

template <class T=real, class A> 
Matrix<T,2,2> to_rot_matrix(const A &ang);

// Conversion to angle

template <class A=real, class T>
A angle(const UnitComplex<T> &c);

template <class A=real, class T>
A angle(const Matrix<T,2,2> &m);

// Complex as rotation representation 

template <class T>
bool is_identity(const UnitComplex<T> &c);

template <class T>
const UnitComplex<T> &normalize(const UnitComplex<T> &c);

}}} // namspace s3d::math::r2

#include "rotation.hpp"

#endif

// $Id: rotation.h 2796 2010-06-28 13:16:00Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

