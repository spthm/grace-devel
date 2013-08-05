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

#ifndef S3D_MATH_FWD_H
#define S3D_MATH_FWD_H

namespace s3d { namespace math {

#ifdef BOOST_SERIALIZATION_LIBRARY_VERSION
#	define HAS_SERIALIZATION 1
#endif

#ifdef REAL_TYPE
typedef REAL_TYPE real;
#else
typedef float real;
#endif

// cannot be enum because of bug #45012
static const int RUNTIME = -1;

template <class T, int D> struct Box;
template <class T, int D=RUNTIME> struct Size;
template <class T, int D=RUNTIME> struct Point;
template <class T, int D=RUNTIME> struct Vector;
template <class T, int D=RUNTIME> struct VectorView;
template <class T, int D=RUNTIME> struct UnitVector;
template <class T, int M=RUNTIME, int N=M> struct Matrix;
template <class T, int D=RUNTIME> struct ParamCoord;

template <class T, int D=RUNTIME> struct Ray;

template <class T, int D> struct Transform;
template <class T, int D> struct LinearTransform;
template <class T, int D> struct AffineTransform;
template <class T, int D> struct ProjTransform;

template <class T, int D> struct Cone;
template <class T, int D> struct Plane;

template <class F> class triangle_view; 
template <class F> class edge_view;

template <class F> class edge_traversal;
template <class F> class triangle_traversal;

enum ortho_plane
{
	PLANE_X,
	PLANE_Y,
	PLANE_Z
};

template <class T> struct is_vector;
template <class T> struct is_point;
template <class T> struct is_size;
template <class T> struct is_transform;
template <class T> struct is_matrix;

} // namespace math

using math::real;
using math::Vector;
using math::Point;
using math::Matrix;
using math::Size;
using math::is_vector;
using math::is_point;
using math::is_size;
using math::is_transform;
using math::is_matrix;

} // namespace s3d

#endif

// $Id: fwd.h 3137 2010-09-21 03:37:15Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

