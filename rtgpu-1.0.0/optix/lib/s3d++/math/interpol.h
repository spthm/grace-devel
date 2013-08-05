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

#ifndef S3D_MATH_INTERPOL_H
#define S3D_MATH_INTERPOL_H

#include "fwd.h"

namespace s3d { namespace math {

template <class T> T lerp(real t, const T &a, const T &b);
template <class T> T bilerp(real t, real ta, const T &a0, const T &a1, 
							        real tb, const T &b0, const T &b1);

template <class T> T bezier(real t, const T &a, const T &b);
template <class T> T bezier(real t, const T &a, const T &b, const T &c);
template <class T> T bezier(real t, const T &a, const T &b, 
									const T &c, const T &d);
}} // s3d::math

#include "interpol.hpp"

#endif


// $Id: interpol.h 2316 2009-06-15 13:36:59Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

