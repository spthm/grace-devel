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

#include <boost/foreach.hpp>
#include "point.h"

namespace s3d { namespace math
{

/*{{{ Result types */
template <class T, class P>
struct result_mul<transform_like<T>,point_like<P>>
{
	typedef typename result_mul_dispatch<typename value_type<T,2>::type,
									     typename value_type<P>::type>::type 
				subtype;

	typedef typename rebind<P,subtype>::type type;

	template <class X=int>
	static inline auto call(const T &t, const P &p)
		-> typename std::enable_if<sizeof(X) && P::dim != T::dim+1, type>::type
	{
		return t.do_multiply_impl(p);
	}

	template <class X=int>
	static inline auto call(const T &t, const P &p)
		-> typename std::enable_if<sizeof(X) && P::dim == T::dim+1, type>::type
	{
		return t.direct() * p;
	}
};

template <class T, class V>
struct result_mul<transform_like<T>,vector_like<V>>
{
	typedef typename result_mul_dispatch<typename value_type<T,2>::type,
									     typename value_type<V>::type>::type
				subtype;

	typedef typename rebind<V,subtype>::type type;

	template <class X=int>
	static inline auto call(const T &t, const V &v)
		-> typename std::enable_if<sizeof(X) && V::dim != T::dim+1, type>::type
	{
		return t.do_multiply_impl(v);
	}

	template <class X=int>
	static inline auto call(const T &t, const V &v)
		-> typename std::enable_if<sizeof(X) && V::dim == T::dim+1, type>::type
	{
		return t.direct() * v;
	}
};

template <class T, class S>
struct result_mul<transform_like<T>,size_like<S>>
{
	typedef typename result_mul_dispatch<typename value_type<T,2>::type,
									     typename value_type<S>::type>::type 
				subtype;

	typedef typename rebind<S,subtype>::type type;

	static inline type call(const T &t, const S &s)
	{
		return t.do_multiply_impl(s);
	}
};

template <class T, class U>
struct result_mul<transform_like<T>,transform_like<U>>
{
	static_assert(T::dim == U::dim, "Mismatched dimensions");

	typedef typename result_mul_dispatch<typename value_type<T,2>::type,
									     typename value_type<U,2>::type>::type 
				subtype;

	typedef typename std::conditional
	<
		(T::xform_order < U::xform_order),
		typename U::template rebind<subtype>,
		typename T::template rebind<subtype>
	>::type::type type;
};

template <class T, class M>
struct result_mul<transform_like<T>, matrix_like<M>>
{
	static_assert(
		// linear_xform
		(T::xform_order == 1 && T::dim==M::dim_rows && T::dim==M::dim_cols) ||
		// affine_xform
		(T::xform_order == 2 && (T::dim==M::dim_rows || T::dim==M::dim_rows+1)
		                     && T::dim==M::dim_cols) ||
		// proj_xform
		(T::xform_order == 3 && (T::dim==M::dim_rows || T::dim==M::dim_rows+1)
		                     && (T::dim==M::dim_cols || T::dim==M::dim_cols+1)),
				   "Mismatched dimensions");

	typedef typename result_mul_dispatch<typename value_type<T,2>::type,
									     typename value_type<M,2>::type>::type 
				subtype;

	typedef typename T::template rebind<subtype>::type type;
};

template <class T, class M>
struct result_mul<matrix_like<M>,transform_like<T>>
	: result_mul<transform_like<T>, matrix_like<M>> // só pra checar constraints
{
	typedef typename T::template rebind<typename result_mul::subtype>::type type;
};
/*}}}*/

template <class T, int D>
Transform<T,D>::Transform(const std::initializer_list<Vector<T,D+1>> &rows)/*{{{*/
	: m_direct(new Matrix<T,D+1,D+1>(rows))
{
}/*}}}*/

template <class T, int D> 
Transform<T,D>::Transform(const Transform &that)/*{{{*/
	: m_direct(new Matrix<T,D+1,D+1>(*that.m_direct))
{
}/*}}}*/

template <class T, int D> 
Transform<T,D>::Transform(Transform &&that)/*{{{*/
{
	swap(m_direct, that.m_direct);
	that.modified();
}/*}}}*/

template <class T, int D> 
Transform<T,D> &Transform<T,D>::operator=(const Transform &that)/*{{{*/
{
	if(that.m_direct)
		m_direct.reset(new Matrix<T,D+1,D+1>(*that.m_direct));
	else
		m_direct.reset();

	if(that.m_inverse)
		m_inverse = that.m_inverse->clone();
	else
		m_inverse.reset();

	if(that.m_transpose)
		m_transpose = that.m_transpose->clone();
	else
		m_transpose.reset();

	return *this;
}/*}}}*/

template <class T, int D> 
Transform<T,D> &Transform<T,D>::operator=(Transform &&that)/*{{{*/
{
	swap(m_direct, that.m_direct);

	m_inverse = std::move(that.m_inverse);
	m_transpose = std::move(that.m_transpose);

	return *this;
}/*}}}*/

template <class T, int D> 
bool Transform<T,D>::operator==(const Transform &that) const/*{{{*/
{
	if(!*this && !that)
		return true;
	else
		return direct() == that.direct();
}/*}}}*/

template <class T, int D>
const Transform<T,D> &Transform<T,D>::inverse() const/*{{{*/
{
	if(*this)
	{
		if(!m_inverse)
			m_inverse.reset(this->do_get_inverse());

		return *m_inverse;
	}
	else
	{
		// Must be duplicated in Transform::transpose because this code doesn't
		// get inlined due to static local variables
		static std::unique_ptr<Transform> ident;
		if(!ident)
		{
			ident = clone();
			set_identity(*ident);
		}
		return *ident;
	}
}/*}}}*/

template <class T, int D>
const Transform<T,D> &Transform<T,D>::transpose() const/*{{{*/
{
	if(*this)
	{
		if(!m_transpose)
		{
			std::unique_ptr<Transform> trans(this->do_get_transpose());

			if(!m_transpose)
				m_transpose = std::move(trans);
		}

		return *m_transpose;
	}
	else
	{
		static std::unique_ptr<Transform> ident;
		if(!ident)
		{
			std::unique_ptr<Transform> id = clone();
			set_identity(*id);

			if(!ident)
				ident = std::move(id);
		}
		return *ident;
	}
}/*}}}*/

template <class T, int D>
void Transform<T,D>::reset()/*{{{*/
{
	m_direct.reset();
	modified();
}/*}}}*/

template <class T, int D> 
Matrix<T,D+1,D+1> &Transform<T,D>::direct()/*{{{*/
{
	if(!m_direct)
		m_direct.reset(new Matrix<T,D+1,D+1>(math::identity<T, D+1>()));
	return *m_direct;
}/*}}}*/
template <class T, int D> 
const Matrix<T,D+1,D+1> &Transform<T,D>::direct() const/*{{{*/
{
	if(m_direct)
		return *m_direct;
	else
	{
		static Matrix<T,D+1,D+1> id = math::identity<T,D+1>();
		return id;
	}
}/*}}}*/

template <class T, int D, template <class,int> class XF> 
void transform_inplace(const XF<T,D> &xf, Point<T,D> &p)/*{{{*/
{
	p = xf * p;
}/*}}}*/

template <class T, int D, template <class,int> class XF> 
void transform_inplace(const XF<T,D> &xf, Vector<T,D> &v)/*{{{*/
{
	v = xf * v;
}/*}}}*/

template <class T, int D, template <class,int> class XF, 
		  template<class,int> class V, class U>
auto homo_transform(const XF<T,D> &xf, const V<U,D> &b)/*{{{*/
	-> typename std::enable_if<is_vector<V<U,D>>::value,
			decltype(xf * augment(b,0))>::type
{
	return xf * augment(b, 0);
}/*}}}*/

template <class T, int D, template <class,int> class XF, 
		  template<class,int> class P, class U>
auto homo_transform(const XF<T,D> &xf, const P<U,D> &b)/*{{{*/
	-> typename std::enable_if<is_point<P<U,D>>::value,
			decltype(xf * augment(b,1))>::type
{
	return xf * augment(b, 1);
}/*}}}*/

namespace detail_math/*{{{*/
{
	template <int N>
	struct xform_dim
	{
		template <class T, int D, template <class,int> class XF> 
		static void xform(Box<T,D> &res, Point<T,D> pt, const Box<T,D> &b, const XF<T,D> &xf)
		{
			res.merge(xf * pt);
			xform_dim<N-1>::xform(res, pt, b, xf);
			
			pt[N-1] += b.size[N-1];
			res.merge(xf * pt);
			xform_dim<N-1>::xform(res, pt, b, xf);
		}
	};
	template <>
	struct xform_dim<0>
	{
		template <class...ARGS>
		static void xform(ARGS...) {}
	};
};/*}}}*/

template <class T, int D, template <class,int> class XF> 
Box<T,D> operator *(const XF<T,D> &xf, const Box<T,D> &b)/*{{{*/
{
	assert(b.is_positive());

	Box<T,D> r(Point<T,D>(0), Size<T,D>(0));

	detail_math::xform_dim<D>::xform(r, b.origin, b, xf);

	assert(r.is_positive());
	return std::move(r);
}/*}}}*/

template <class T, int D, template <class,int> class XF> 
void transform_inplace(const XF<T,D> &xf, Box<T,D> &b)/*{{{*//*{{{*/
{
	b = xf * b;
}/*}}}*//*}}}*/

template <class T, int D, template <class,int> class XF> 
Ray<T,D> operator *(const XF<T,D> &xf, const Ray<T,D> &r)/*{{{*/
{
	return Ray<T,D>(xf * r.origin, xf * r.dir);
}/*}}}*/

template <class T, int D, template <class,int> class XF> 
void transform_inplace(const XF<T,D> &xf, Ray<T,D> &r)/*{{{*/
{
	transform_inplace(xf, r.origin);
	transform_inplace(xf, r.dir);
}/*}}}*/

}}

// $Id: transform.hpp 3137 2010-09-21 03:37:15Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

