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

#ifndef S3D_MATH_AFFINE_TRANSFORM_H
#define S3D_MATH_AFFINE_TRANSFORM_H

#include "linear_transform.h"
#include "matrix.h"
#include "vector.h"
#include "size.h"

namespace s3d { namespace math
{

template <class T, int D> class AffineTransform
	: public LinearTransform<T,D>
{
	typedef LinearTransform<T,D> base;
public:
	static const int xform_order = 2;

	template <class U, int P=D> 
	struct rebind { typedef AffineTransform<U,P> type; };

	AffineTransform() {}
	AffineTransform(const AffineTransform &xf);
	AffineTransform(AffineTransform &&xf);

	AffineTransform(const std::initializer_list<Vector<T,D+1>> &rows);

	template <int M, int N>
	explicit AffineTransform(const Matrix<T,M,N> &m);

	explicit AffineTransform(const Matrix<T,D,D> &m) : base(m) {}
	explicit AffineTransform(const Size<T,D> &s) : base(s) {}
	explicit AffineTransform(const Vector<T,D> &v);

	template <class XF, class = typename std::enable_if<
		(XF::xform_order < xform_order) && XF::dim==D>::type>
	AffineTransform(const XF &xf);

	template <class XF, class = typename std::enable_if<
		(XF::xform_order < xform_order) && XF::dim==D>::type>
	AffineTransform(XF &&xf);

	template <class XF, class = typename std::enable_if<
		(XF::xform_order < xform_order) && XF::dim==D>::type>
	AffineTransform &operator=(const XF &xf);

	template <class XF, class = typename std::enable_if<
		(XF::xform_order < xform_order) && XF::dim==D>::type>
	AffineTransform &operator=(XF &&xf);

	AffineTransform &operator=(const AffineTransform &xf);
	AffineTransform &operator=(AffineTransform &&xf);

    AffineTransform &operator*=(const AffineTransform &xf);

	template <int M, int N>
    AffineTransform &operator*=(const Matrix<T,M,N> &a);

	AffineTransform &operator*=(const Size<T,D> &s);
	AffineTransform &operator/=(const Size<T,D> &s);

	AffineTransform &operator+=(const Vector<T,D> &s);
	AffineTransform &operator-=(const Vector<T,D> &s);

	AffineTransform &operator*=(const T &s);
	AffineTransform &operator/=(const T &s);

	template <class... ARGS> 
	AffineTransform &translate_by(const ARGS &...v);
	AffineTransform &translate_by(const Vector<T,D> &v);

	template <class... ARGS> 
	AffineTransform &scale_by(const ARGS &...v);

	template <class... ARGS> 
	AffineTransform &rotate_by(const ARGS &...v);

	friend const AffineTransform &inv(const AffineTransform &xf)/*{{{*/
	{
		return static_cast<const AffineTransform &>(xf.inverse());
	}/*}}}*/
	friend const AffineTransform &transpose(const AffineTransform &xf)/*{{{*/
	{
		return static_cast<const AffineTransform &>(xf.transpose());
	}/*}}}*/

	DEFINE_CLONABLE(AffineTransform);
	DEFINE_MOVABLE(AffineTransform);

protected:
	AffineTransform(typename base::init_base_xform_tag tag,
					const std::initializer_list<Vector<T,D+1>> &rows)
		: base(tag,rows) {}

	using base::do_multiply_impl;

	template <class P>
	auto do_multiply_impl(const P &p) const
		-> typename std::enable_if<is_point<P>::value, P>::type;

private:
	template <class,class> friend struct result_mul;

	virtual AffineTransform *do_get_inverse() const;
	virtual AffineTransform *do_get_transpose() const;
	virtual Point<T,D> do_multiply(const Point<T,D> &p) const
		{ return do_multiply_impl(p); }
};

template <class T, int D>
struct is_transform<AffineTransform<T,D>>
{
	static const bool value = true;
};

template <class T>
AffineTransform<T,3> look_at(const Point<T,3> &eye,
						     const Point<T,3> &center,
						     const UnitVector<T,3> &up);

}} // namespace s3d::math

#include "affine_transform.hpp"

#endif

// $Id: affine_transform.h 3137 2010-09-21 03:37:15Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

