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

#ifndef S3D_MATH_TRANSFORM_H
#define S3D_MATH_TRANSFORM_H

#include "../util/optional.h"
#include "../util/unique_ptr.h"
#include "../util/clonable.h"
#include "../util/movable.h"
#include "fwd.h"
#include "size.h"
#include "matrix.h"

namespace s3d { namespace math
{

template <class T>
struct is_transform
{
	static const bool value = false;
};

template <class T>
struct is_transform<T&> : is_transform<typename std::remove_cv<T>::type>
{
};

template <class T, int D> class Transform 
	: public clonable
	, public movable
	, public operators
{
	struct safe_bool { int a; };
public:
	static_assert(D>=0, "Invalid transform dimension");

	typedef Vector<T,D+1> value_type;
	static const size_t dim = D;

	Transform() {}

	Transform(const std::initializer_list<Vector<T,D+1>> &rows);
	Transform(const Matrix<T,D+1,D+1> &m) 
		: m_direct(new Matrix<T,D+1,D+1>(m)) {}

	const Vector<T,D+1> operator[](int i) const { return direct()[i]; }

	bool operator==(const Transform &that) const;
	bool operator!=(const Transform &that) const { return !operator==(that); }

	void update_related();
	friend bool is_identity(const Transform &xform)
		{ return xform.m_direct ? is_identity(*xform.m_direct) : true; }

	operator int safe_bool::*() const { return m_direct?&safe_bool::a : NULL; }
	bool operator!() const { return m_direct ? false : true; }

	operator const Matrix<T,D+1,D+1> &() const { return direct(); }

	template <class U>
	operator Matrix<U,D+1,D+1> () const { return direct(); }

	// Must be defined here due to language constraints
	friend const Transform &inv(const Transform &xf)/*{{{*/
	{
		return xf.inverse();
	}/*}}}*/
	friend const Transform &transpose(const Transform &xf)/*{{{*/
	{
		return xf.transpose();
	}/*}}}*/
	friend void set_identity(Transform &xf)/*{{{*/
	{
		xf.reset();
	}/*}}}*/

	DEFINE_PURE_CLONABLE(Transform);
	DEFINE_PURE_MOVABLE(Transform);

	friend std::ostream &operator<<(std::ostream &out, const Transform &xf)/*{{{*/
	{
		return out << static_cast<const Matrix<T,D+1,D+1> &>(xf);
	}/*}}}*/

protected:
	Transform(const Transform &that);
	Transform(Transform &&that);
	Transform &operator=(const Transform &that);
	Transform &operator=(Transform &&that);

	const Transform &inverse() const;
	const Transform &transpose() const;
	void reset();
	void modified() { m_inverse.reset(); m_transpose.reset(); }

	Matrix<T,D+1,D+1> &direct();
	const Matrix<T,D+1,D+1> &direct() const;

	template <class,class> friend struct result_mul;

private:
	virtual Point<T,D> do_multiply(const Point<T,D> &p) const = 0;
	virtual Vector<T,D> do_multiply(const Vector<T,D> &p) const = 0;
	virtual Size<T,D> do_multiply(const Size<T,D> &p) const = 0;

	virtual Transform *do_get_inverse() const = 0;
	virtual Transform *do_get_transpose() const = 0;

	// Is null on identity transform
	mutable std::unique_ptr<Matrix<T,D+1,D+1>> m_direct;

	// Only gets calculated when needed.
	// Gets reset when m_direct is changed
	mutable std::unique_ptr<Transform> m_inverse;
	mutable std::unique_ptr<Transform> m_transpose;
};

template <class T, int N>
struct is_transform<Transform<T,N>>
{
	static const bool value = true;
};

template <class T>
struct transform_like : concept<T> {};

template <class T, int D, template <class,int> class XF> 
void transform_inplace(const XF<T,D> &xf, Point<T,D> &b);

template <class T, int D, template <class,int> class XF> 
void transform_inplace(const XF<T,D> &xf, Vector<T,D> &b);

template <class T, int D, template <class,int> class XF> 
Box<T,D> operator *(const XF<T,D> &xf, const Box<T,D> &b);

template <class T, int D, template <class,int> class XF> 
void transform_inplace(const XF<T,D> &xf, Box<T,D> &b);

template <class T, int D, template <class,int> class XF> 
Ray<T,D> operator *(const XF<T,D> &xf, const Ray<T,D> &r);

template <class T, int D, template <class,int> class XF> 
void transform_inplace(const XF<T,D> &xf, Ray<T,D> &r);

template <class T, int D> 
Matrix<T,D+1,D+1> operator*(const Transform<T,D> &t1, const Transform<T,D> &t2);

}

template <class T>
struct concept_arg<T, typename std::enable_if<math::is_transform<T>::value>::type>
{
	typedef math::transform_like<T> type;
};

} // s3d::math

#include "transform.hpp"

#endif

// $Id: transform.h 3137 2010-09-21 03:37:15Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

