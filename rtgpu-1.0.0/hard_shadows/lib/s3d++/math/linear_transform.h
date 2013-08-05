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

#ifndef S3D_MATH_LINEAR_TRANSFORM_H
#define S3D_MATH_LINEAR_TRANSFORM_H

#include "rotation.h"
#include "transform.h"
#include "matrix.h"
#include "vector.h"
#include "size.h"

namespace s3d { namespace math
{

template <class T, int D> class LinearTransform
	: public Transform<T,D>
{
	typedef Transform<T,D> base;
public:
	template <class U, int P=D> 
	struct rebind { typedef LinearTransform<U,P> type; };

	LinearTransform() {}
	LinearTransform(const LinearTransform &xf);
	LinearTransform(LinearTransform &&xf);
	template <int M, int N>
	explicit LinearTransform(const Matrix<T,M,N> &m);
	LinearTransform(const std::initializer_list<Vector<T,D>> &rows);

	static const int xform_order = 1;

	typedef Vector<T,D+1> value_type;

	explicit LinearTransform(const Size<T,D> &s);

	using base::operator[];
	value_type &operator[](int i);
	const value_type &operator[](int i) const;

	LinearTransform &operator=(const LinearTransform &xf);
	LinearTransform &operator=(LinearTransform &&xf);

    LinearTransform &operator*=(const LinearTransform &xf);
    LinearTransform &operator*=(const Matrix<T,D,D> &a);

	LinearTransform &operator*=(const Size<T,D> &s);
	LinearTransform &operator/=(const Size<T,D> &s);

	LinearTransform &operator*=(const T &s);
	LinearTransform &operator/=(const T &s);

	template <class... ARGS> 
	LinearTransform &scale_by(const ARGS &...v);
	LinearTransform &scale_by(const Size<T,D> &v);

	template <class U>
	LinearTransform &rotate_by(const U &r);
	LinearTransform &rotate_by(const UnitVector<T,D> &axis, T angle);

	friend const LinearTransform &inv(const LinearTransform &xf)/*{{{*/
	{
		return static_cast<const LinearTransform &>(xf.inverse());
	}/*}}}*/
	friend const LinearTransform &transpose(const LinearTransform &xf)/*{{{*/
	{
		return static_cast<const LinearTransform &>(xf.transpose());
	}/*}}}*/

	DEFINE_CLONABLE(LinearTransform);
	DEFINE_MOVABLE(LinearTransform);

protected:
	struct init_base_xform_tag {};
	LinearTransform(init_base_xform_tag tag,
					const std::initializer_list<Vector<T,D+1>> &rows)
		: Transform<T,D>(rows) {}

	LinearTransform(init_base_xform_tag tag,
					const Matrix<T,4,4> &m)
		: Transform<T,D>(m) {}

protected:
	template <class,class> friend struct result_mul;
	template <class P> auto do_multiply_impl(const P &p) const
		-> typename std::enable_if<is_point<P>::value, P>::type;

	template <class V> auto do_multiply_impl(const V &p) const
		-> typename std::enable_if<is_vector<V>::value || 
								   is_size<V>::value, V>::type;

private:
	virtual Point<T,D> do_multiply(const Point<T,D> &p) const
		{ return do_multiply_impl(p); }
	virtual Vector<T,D> do_multiply(const Vector<T,D> &p) const
		{ return do_multiply_impl(p); }
	virtual Size<T,D> do_multiply(const Size<T,D> &p) const
		{ return do_multiply_impl(p); }

	virtual LinearTransform *do_get_inverse() const;
	virtual LinearTransform *do_get_transpose() const;
};

template <class T, int D>
struct is_transform<LinearTransform<T,D>>
{
	static const bool value = true;
};

template <class T>
ProjTransform<T,3> persp_proj(const Box<T,2> &wnd, T near, T far);

template <class T>
ProjTransform<T,3> persp_proj(T fovy, T aspect, T near, T far);

template <class T>
AffineTransform<T,3> ortho_proj(const Box<T,2> &wnd, T near, T far);

}} // namespace s3d::math

#include "linear_transform.hpp"

#endif

// $Id: linear_transform.h 3205 2010-11-15 02:23:59Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

