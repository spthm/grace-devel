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

#include "point.h"

namespace s3d { namespace math
{

template <class T, int D> 
AffineTransform<T,D>::AffineTransform(const AffineTransform &xf)/*{{{*/
{
	*this = xf;
}/*}}}*/

template <class T, int D> 
AffineTransform<T,D>::AffineTransform(AffineTransform &&xf)/*{{{*/
{
	*this = std::move(xf);
}/*}}}*/

template <class T, int D> template <int M, int N> 
AffineTransform<T,D>::AffineTransform(const Matrix<T,M,N> &m)/*{{{*/
{
	static_assert(M>=D && N>=D+1, "Matrix dimension must be >= D x D+1");

	Matrix<T,D+1,D+1> &td = this->direct();

	for(int i=0; i<D; ++i)
		std::copy(m[i].begin(), m[i].begin() + D+1, td[i].begin());

	td[D] = augment(Vector<T,D>(0), 1);
}/*}}}*/

template <class T, int D>
AffineTransform<T,D>::AffineTransform(const std::initializer_list<Vector<T,D+1>> &rows)/*{{{*/
	: LinearTransform<T,D>(identity_augment<D+1,D+1>(Matrix<T,D,D+1>(rows)))
{
}/*}}}*/

template <class T, int D> 
AffineTransform<T,D>::AffineTransform(const Vector<T,D> &v)/*{{{*/
{
	// [1 0 x]
	// [0 1 y]
	// [0 0 1]

	Matrix<T,D+1,D+1> &td = this->direct();

	for(int i=0; i<D; ++i)
		td[i][D] = v[i];
}/*}}}*/

template <class T, int D> template <class XF, class>
AffineTransform<T,D>::AffineTransform(const XF &xf)/*{{{*/
{
	*this = xf;
}/*}}}*/

template <class T, int D> template <class XF, class>
AffineTransform<T,D>::AffineTransform(XF &&xf)/*{{{*/
{
	*this = std::move(xf);
}/*}}}*/

template <class T, int D> template <class XF, class>
AffineTransform<T,D> &AffineTransform<T,D>::operator=(const XF &xf)/*{{{*/
{
	Transform<T,D>::operator=(xf);

	if(typeid(xf) != typeid(*this))
	{
		static Vector<T,D+1> projrow = augment(Vector<T,D>(0),1);
		this->direct()[D] = projrow;
	}

	return *this;
}/*}}}*/

template <class T, int D> template <class XF, class>
AffineTransform<T,D> &AffineTransform<T,D>::operator=(XF &&xf)/*{{{*/
{
	Transform<T,D>::operator=(std::move(xf));

	if(typeid(xf) != typeid(*this))
	{
		static Vector<T,D+1> projrow = augment(Vector<T,D>(0),1);
		this->direct()[D] = projrow;
	}

	return *this;
}/*}}}*/

template <class T, int D>
AffineTransform<T,D> &AffineTransform<T,D>::operator=(const AffineTransform &xf)/*{{{*/
{
	std::copy(xf.direct().begin(), xf.direct().begin()+D, this->direct().begin());
	this->modified();
	return *this;
}/*}}}*/
template <class T, int D>
AffineTransform<T,D> &AffineTransform<T,D>::operator=(AffineTransform &&xf)/*{{{*/
{
	base::operator=(std::move(xf));
	std::copy(xf.direct().begin(), xf.direct().begin()+D, this->direct().begin());
	this->modified();
	return *this;
}/*}}}*/

template <class T, int D> 
AffineTransform<T,D> &AffineTransform<T,D>::operator*=(const AffineTransform &xf)/*{{{*/
{
	//   [a1.a2+b1.c2  b1.d2+a1.b2  x1+a1.x2+b1.y2]
	//	 [c1.a2+d1.c2  d1.d2+c1.b2  y1+c1.x2+d1.y2]
	//	 [     0            0               1     ]

	if(xf)
	{
		if(*this)
		{
			Matrix<T,D+1,D+1> &td = this->direct();

			for(int i=0; i<D; ++i)
			{
				Vector<T,D+1> r;
				for(int j=0; j<D+1; ++j)
				{
					r[j] = td[i][0]*xf[0][j];
					for(int k=1; k<D; ++k)
						r[j] += td[i][k]*xf[k][j];
				}
				r[D] += td[i][D];
				td[i] = r;
			}
			this->modified();
		}
		else
			*this = xf;
	}
	return *this;
}/*}}}*/

template <class T, int D> template <int M, int N>
AffineTransform<T,D> &AffineTransform<T,D>::operator*=(const Matrix<T,M,N> &a)/*{{{*/
{
	base::operator*=(a);
	return *this;
}/*}}}*/

template <class T, int D> 
AffineTransform<T,D> &AffineTransform<T,D>::operator*=(const Size<T,D> &s)/*{{{*/
{
	base::operator*=(s);
	return *this;
}/*}}}*/
template <class T, int D> 
AffineTransform<T,D> &AffineTransform<T,D>::operator/=(const Size<T,D> &s)/*{{{*/
{
	base::operator*=(s);
	return *this;
}/*}}}*/

template <class T, int D> 
AffineTransform<T,D> &AffineTransform<T,D>::operator*=(const T &s)/*{{{*/
{
	base::operator*=(s);
	return *this;
}/*}}}*/
template <class T, int D> 
AffineTransform<T,D> &AffineTransform<T,D>::operator/=(const T &s)/*{{{*/
{
	base::operator*=(s);
	return *this;
}/*}}}*/

template <class T, int D> 
AffineTransform<T,D> &AffineTransform<T,D>::operator+=(const Vector<T,D> &v)/*{{{*/
{
	translate_by(v);
	return *this;
}/*}}}*/
template <class T, int D> 
AffineTransform<T,D> &AffineTransform<T,D>::operator-=(const Vector<T,D> &v)/*{{{*/
{
	translate_by(-v);
	return *this;
}/*}}}*/

template <class T, int D> template <class P>
auto AffineTransform<T,D>::do_multiply_impl(const P &p) const/*{{{*/
	-> typename std::enable_if<is_point<P>::value, P>::type
{
	// [a.px + b.py + x]
	// [c.px + d.py + y]
	// [       1       ]

	const Matrix<T,D+1,D+1> &td = this->direct();

	static_assert(P::dim == D, "Mismatched dimensions");
	if(p.size()+1 != td.rows())
		throw std::runtime_error("Mismatched dimensions");

	P ret;

	for(int i=0; i<D; ++i)
	{
		ret[i] = td[i][D];
		for(int j=0; j<D; ++j)
			ret[i] += td[i][j]*p[j];
	}

	return ret;
}/*}}}*/

template <class T, int D> 
auto AffineTransform<T,D>::translate_by(const Vector<T,D> &v)/*{{{*/
	-> AffineTransform &
{
	// [a  b  x+a.vx+b.vy]
	// [c  d  y+c.vx+d.vy]
	// [0  0       1     ]

	Matrix<T,D+1,D+1> &td = this->direct();
	for(int i=0; i<D; ++i)
		for(int j=0; j<D; ++j)
			td[i][D] += td[i][j]*v[j];

	this->modified();
	return *this;
}/*}}}*/

template <class T, int D> template <class... ARGS> 
auto AffineTransform<T,D>::translate_by(const ARGS &...v)/*{{{*/
	-> AffineTransform &
{
	return translate_by(Vector<T,D>(v...));
}/*}}}*/

template <class T, int D> template <class... ARGS> 
auto AffineTransform<T,D>::scale_by(const ARGS &...v)/*{{{*/
	-> AffineTransform &
{
	base::scale_by(v...);
	return *this;
}/*}}}*/

template <class T, int D> template <class... ARGS> 
auto AffineTransform<T,D>::rotate_by(const ARGS &...v)/*{{{*/
	-> AffineTransform &
{
	base::rotate_by(v...);
	return *this;
}/*}}}*/

template <class T, int D>
AffineTransform<T,D> *AffineTransform<T,D>::do_get_inverse() const/*{{{*/
{
	return new AffineTransform<T,D>(
		submatrix<subrows<D>>(inv(this->direct())));
}/*}}}*/

template <class T, int D>
AffineTransform<T,D> *AffineTransform<T,D>::do_get_transpose() const/*{{{*/
{
	return new AffineTransform<T,D>(
		submatrix<subrows<D>>(transpose(this->direct())));
}/*}}}*/

template <class T>
AffineTransform<T,3> look_at(const Point<T,3> &eye,
						   const Point<T,3> &center,
						   const UnitVector<T,3> &up)
{
	auto v = unit(center-eye);
	auto s = cross(v,up),
		 u = cross(s,v);

	return AffineTransform<T,3>(origin<T,3>()-eye)*LinearTransform<T,3>{s,u,-v};
}

}}

// $Id: affine_transform.hpp 3137 2010-09-21 03:37:15Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

