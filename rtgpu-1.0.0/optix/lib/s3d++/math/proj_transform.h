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

#ifndef S3D_MATH_PROJ_TRANSFORM_H
#define S3D_MATH_PROJ_TRANSFORM_H

#include "affine_transform.h"
#include "matrix.h"
#include "vector.h"
#include "size.h"

namespace s3d { namespace math
{

template <class T, int D> class ProjTransform
	: public AffineTransform<T,D>
{
	typedef AffineTransform<T,D> base;
public:
	static const int xform_order = 3;

	template <class U, int P=D> 
	struct rebind { typedef ProjTransform<U,P> type; };

	ProjTransform() {}
	ProjTransform(const ProjTransform &xf);
	ProjTransform(ProjTransform &&xf);

	template <class XF, class = typename std::enable_if<
		(XF::xform_order < xform_order) && XF::dim==D>::type>
	ProjTransform(const XF &xf);

	template <class XF, class = typename std::enable_if<
		(XF::xform_order < xform_order) && XF::dim==D>::type>
	ProjTransform(XF &&xf);

	ProjTransform(const std::initializer_list<Vector<T,D+1>> &rows);

	template <int M, int N>
	explicit ProjTransform(const Matrix<T,M,N> &m);
	explicit ProjTransform(const Size<T,D> &s) : base(s) {}
	explicit ProjTransform(const Vector<T,D> &v) : base(v) {}

	template <class XF, class = typename std::enable_if<
		(XF::xform_order < xform_order) && XF::dim==D>::type>
	ProjTransform &operator=(const XF &xf);

	template <class XF, class = typename std::enable_if<
		(XF::xform_order < xform_order) && XF::dim==D>::type>
	ProjTransform &operator=(XF &&xf);

	ProjTransform &operator=(const ProjTransform &xf);
	ProjTransform &operator=(ProjTransform &&xf);

	using base::operator[];
	Vector<T,D+1> &operator[](int i);

    ProjTransform &operator*=(const ProjTransform &xf);

	template <int M, int N>
    ProjTransform &operator*=(const Matrix<T,M,N> &a);

	ProjTransform &operator*=(const Size<T,D> &s);
	ProjTransform &operator/=(const Size<T,D> &s);

	ProjTransform &operator+=(const Vector<T,D> &s);
	ProjTransform &operator-=(const Vector<T,D> &s);

	ProjTransform &operator*=(const T &s);
	ProjTransform &operator/=(const T &s);

	template <class... ARGS> ProjTransform &translate_by(const ARGS &...v);
	ProjTransform &translate_by(const Vector<T,D> &v);

	template <class... ARGS> ProjTransform &scale_by(const ARGS &...v);
	ProjTransform &scale_by(const Size<T,D> &s);

	template <class U> ProjTransform &rotate_by(const U &r);
	ProjTransform &rotate_by(const UnitVector<T,D> &axis, T angle);

	friend const ProjTransform &inv(const ProjTransform &xf)/*{{{*/
	{
		return static_cast<const ProjTransform &>(xf.inverse());
	}/*}}}*/
	friend const ProjTransform &transpose(const ProjTransform &xf)/*{{{*/
	{
		return static_cast<const ProjTransform &>(xf.transpose());
	}/*}}}*/

	DEFINE_CLONABLE(ProjTransform);
	DEFINE_MOVABLE(ProjTransform);

protected:
	template <class,class> friend struct result_mul;
	using base::do_multiply_impl;

	template <class P> auto do_multiply_impl(const P &p) const
		-> typename std::enable_if<is_point<P>::value, P>::type;
private:
	virtual ProjTransform *do_get_inverse() const;
	virtual ProjTransform *do_get_transpose() const;
	virtual Point<T,D> do_multiply(const Point<T,D> &p) const
		{ return do_multiply_impl(p); }
};

template <class T, int D>
struct is_transform<ProjTransform<T,D>>
{
	static const bool value = true;
};

template <class T>
Point<T,2> proj(const Point<T,3> &p, ortho_plane plane);

template <class T, template <class,int> class V>
auto proj(const V<T,3> &p, math::ortho_plane plane)
	-> typename requires<is_vector<V<T,3>>, Vector<T,2>>::type;

template <class T>
Box<T,2> proj(const Box<T,3> &b, ortho_plane plane);

template <class T>
Size<T,2> proj(const Size<T,3> &s, ortho_plane plane);

template <class T, template <class,int> class P>
Point<T,3> unproj(const Point<T,2> &p, real z, ortho_plane plane);

template <class T, template <class,int> class P>
Vector<T,3> unproj(const Vector<T,2> &p, real z, ortho_plane plane);

template <class T>
Point<T,2> proj(const Point<T,3> &p, const ProjTransform<T,3> &xf);

template <class T>
Box<T,2> proj(const Box<T,3> &p, const ProjTransform<T,3> &xf);

template <class T>
Point<T,3> unproj(const Point<T,2> &p, T z, const ProjTransform<T,3> &xf);

/* Spherical projection:
	  0 <= theta < 2*pi   (rotation around +y)
	-pi <   phi  < pi     (rotation around +x)

	North pole: +y
*/

template <class T>
Point<T,2> proj_sphere(const Point<T,3> &p);

template <class T>
Point<T,3> unproj_sphere(const Point<T,2> &p);

template <class T>
ProjTransform<T,3> persp_proj(const Box<T,2> &wnd, T near, T far);

template <class T>
ProjTransform<T,3> persp_proj(T fovy, T aspect, T near, T far);

inline auto persp_proj(real fovy, real aspect, real near, real far)
	-> ProjTransform<real,3> 
	{ return persp_proj<real>(fovy, aspect, near, far); }

template <class T>
AffineTransform<T,3> ortho_proj(const Box<T,2> &wnd, T near, T far);


}} // namespace s3d::math

#include "proj_transform.hpp"

#endif

// $Id: proj_transform.h 2240 2009-06-05 22:49:41Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

