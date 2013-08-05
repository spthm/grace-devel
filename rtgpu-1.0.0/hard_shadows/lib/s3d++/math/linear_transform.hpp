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


namespace s3d { namespace math
{

static_assert(order<LinearTransform<float,3>>::value == 2, "Erro");

template <class T, int D> 
LinearTransform<T,D>::LinearTransform(const LinearTransform &xf)/*{{{*/
{
	*this = xf;
}/*}}}*/

template <class T, int D> 
LinearTransform<T,D>::LinearTransform(LinearTransform &&xf)/*{{{*/
{
	*this = std::move(xf);
}/*}}}*/

template <class T, int D>
LinearTransform<T,D>::LinearTransform(const std::initializer_list<Vector<T,D>> &rows)/*{{{*/
	: Transform<T,D>(identity_augment<D+1,D+1>(Matrix<T,D,D>(rows)))
{
}/*}}}*/

template <class T, int D> template <int M, int N>
LinearTransform<T,D>::LinearTransform(const Matrix<T,M,N> &m)/*{{{*/
{
	static_assert(M>=D && N>=D, "Matrix dimension must be >= D x D");

	// [a b 0]
	// [c d 0]
	// [0 0 1]

	Matrix<T,D+1,D+1> &td = this->direct();

	for(int i=0; i<D; ++i)
		std::copy(m[i].begin(), m[i].begin()+D, td[i].begin());
}/*}}}*/
template <class T, int D> 
LinearTransform<T,D>::LinearTransform(const Size<T,D> &s)/*{{{*/
{
	// [a 0 0]
	// [0 d 0]
	// [0 0 1]
	
	Matrix<T,D+1,D+1> &td = this->direct();

	for(int i=0; i<D; ++i)
		td[i][i] = s[i];
}/*}}}*/

template <class T, int D> 
LinearTransform<T,D> &LinearTransform<T,D>::operator=(const LinearTransform &xf)/*{{{*/
{
	Matrix<T,D+1,D+1> &td = this->direct();

	for(int i=0; i<D; ++i)
		for(int j=0; j<D; ++j)
			td[i][j] = xf[i][j];

	this->modified();

	return *this;
}/*}}}*/

template <class T, int D> 
LinearTransform<T,D> &LinearTransform<T,D>::operator=(LinearTransform &&xf)/*{{{*/
{
	base::operator=(std::move(xf));

	Matrix<T,D+1,D+1> &td = this->direct();

	if(typeid(xf) != typeid(*this))
	{
		for(int i=0; i<D-1; ++i)
			td[D][i] = td[i][D] = 0;
		td[D][D] = 1;
	}

	return *this;
}/*}}}*/

template <class T, int D> 
auto LinearTransform<T,D>::operator[](int i) -> value_type &/*{{{*/
{
	assert(i>=0 && i<D);
	this->modified();
	return this->direct()[i];
}/*}}}*/

template <class T, int D> 
auto LinearTransform<T,D>::operator[](int i) const -> const value_type &/*{{{*/
{
	assert(i>=0 && i<D);
	return this->direct()[i];
}/*}}}*/

template <class T, int D> 
LinearTransform<T,D> &LinearTransform<T,D>::operator*=(const LinearTransform &xf)/*{{{*/
{
	//   [a1.a2+b1.c2  b1.d2+a1.b2  0]
	//	 [c1.a2+d1.c2  d1.d2+c1.b2  0]
	//	 [     0            0       1]

	if(xf)
	{
		if(*this)
		{
			Matrix<T,D+1,D+1> &td = this->direct();

			for(int i=0; i<D; ++i)
			{
				Vector<T,D> r;
				for(int j=0; j<D; ++j)
				{
					r[j] = td[i][0]*xf[0][j];
					for(int k=1; k<D; ++k)
						r[j] += td[i][k]*xf[k][j];

				}
				std::copy(r.begin(), r.end(), td[i].begin());
			}

			this->modified();
		}
		else
			*this = xf;
	}
	return *this;
}/*}}}*/

template <class T, int D>
LinearTransform<T,D> &LinearTransform<T,D>::operator*=(const Matrix<T,D,D> &a)/*{{{*/
{
	return *this *= LinearTransform(a);
}/*}}}*/

template <class T, int D> 
LinearTransform<T,D> &LinearTransform<T,D>::operator*=(const Size<T,D> &s)/*{{{*/
{
	scale_by(s);
	return *this;
}/*}}}*/
template <class T, int D> 
LinearTransform<T,D> &LinearTransform<T,D>::operator/=(const Size<T,D> &s)/*{{{*/
{
	scale_by(1/s);
	return *this;
}/*}}}*/

template <class T, int D> 
LinearTransform<T,D> &LinearTransform<T,D>::operator*=(const T &s)/*{{{*/
{
	scale_by(s);
	return *this;
}/*}}}*/
template <class T, int D> 
LinearTransform<T,D> &LinearTransform<T,D>::operator/=(const T &s)/*{{{*/
{
	scale_by(1/s);
	return *this;
}/*}}}*/

template <class T, int D> template <class P>
auto LinearTransform<T,D>::do_multiply_impl(const P &p) const/*{{{*/
	-> typename std::enable_if<is_point<P>::value, P>::type
{
	// [a.px + b.py]
	// [c.px + d.py]
	// [     1     ]

	const Matrix<T,D+1,D+1> &td = this->direct();

	static_assert(P::dim == D, "Mismatched dimensions");
	if(p.size()+1 != td.rows())
		throw std::runtime_error("Mismatched dimensions");

	P ret;

	for(int i=0; i<D; ++i)
	{
		ret[i] = td[i][0]*p[0];
		for(int j=1; j<D; ++j)
			ret[i] += td[i][j]*p[j];
	}

	return ret;
}/*}}}*/

template <class T, int D> template <class V>
auto LinearTransform<T,D>::do_multiply_impl(const V &v) const/*{{{*/
	-> typename std::enable_if<is_vector<V>::value || is_size<V>::value,V>::type
{
	const Matrix<T,D+1,D+1> &td = this->direct();

	static_assert(V::dim == D, "Mismatched dimensions");

	if(v.size()+1 != td.rows())
	   throw std::runtime_error("Mismatched dimensions");

	// [a.vx + b.vy]
	// [c.vx + d.vy]
	// [     0     ]

	V ret(dim(v.size()));

	for(int i=0; i<D; ++i)
	{
		ret[i] = td[i][0]*v[0];
		for(int j=1; j<D; ++j)
			ret[i] += td[i][j]*v[j];
	}

	return ret;
}/*}}}*/

template <class T, int D> 
auto LinearTransform<T,D>::scale_by(const Size<T,D> &s)/*{{{*/
	-> LinearTransform &
{
	// [a.sx  b.sy  0]
	// [c.sx  d.sy  0]
	// [ 0     0    1]
	
	Matrix<T,D+1,D+1> &td = this->direct();
	for(int i=0; i<D; ++i)
		for(int j=0; j<D; ++j)
			td[i][j] *= s[j];

	this->modified();

	return *this;
}/*}}}*/

template <class T, int D> template <class... ARGS> 
auto LinearTransform<T,D>::scale_by(const ARGS &...v)/*{{{*/
	-> LinearTransform &
{
	return scale_by(Size<T,D>(v...));
}/*}}}*/

template <class T, int D> template <class U> 
auto LinearTransform<T,D>::rotate_by(const U &r)/*{{{*/
	-> LinearTransform &
{
	using math::r2::to_rot_matrix;
	using math::r3::to_rot_matrix;

	return *this *= to_rot_matrix(r);
}/*}}}*/

template <class T, int D>
auto LinearTransform<T,D>::rotate_by(const UnitVector<T,D> &axis, T angle)/*{{{*/
	-> LinearTransform<T,D> &
{
	return rotate_by(AxisAngle<T>(axis, angle));
}/*}}}*/

template <class T, int D>
LinearTransform<T,D> *LinearTransform<T,D>::do_get_inverse() const/*{{{*/
{
	return new LinearTransform<T,D>(
		inv(submatrix<subrows<D>,subcols<D>>(this->direct())));
}/*}}}*/

template <class T, int D>
LinearTransform<T,D> *LinearTransform<T,D>::do_get_transpose() const/*{{{*/
{
	return new LinearTransform<T,D>(
		transpose(submatrix<subrows<D>,subcols<D>>(this->direct())));
}/*}}}*/

}}

// $Id: linear_transform.hpp 3137 2010-09-21 03:37:15Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

