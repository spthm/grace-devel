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

#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include "point.h"
#include "../util/functor.h"
#include "../util/tuple.h"
#include "../mpl/unique.h"
#include "../mpl/erase.h"
#include "../mpl/exists.h"
#include "../mpl/at.h"
#include "../mpl/int.h"
#include "../mpl/add.h"
#include "../mpl/sub.h"
#include "vector_view.h"

namespace s3d { namespace math
{

static_assert(order<Matrix<float,2,3>>::value==2, "Error in matrix order");
static_assert(std::is_same<value_type<Matrix<float,2,3>,2>::type,
						   float>::value, "Error in matrix value_type");

namespace detail/*{{{*/
{
	template<int...IDX>
	struct equal_dim/*{{{*/
	{
		typedef typename mpl::unique 
		< 
			mpl::erase_c<mpl::vector_c<int,IDX...>,RUNTIME> 
		>::type common;

		static const bool value = common::size<=1;
	};/*}}}*/

	template<int...IDX>
	struct common_dim/*{{{*/
	{
		typedef typename mpl::unique 
		< 
			mpl::erase_c<mpl::vector_c<int,IDX...>,RUNTIME> 
		>::type aux;

		typedef typename std::conditional
		<
			aux::size == 0,
			mpl::vector_c<int,RUNTIME>,
			aux
		>::type common;

		static_assert(common::size == 1, "Inconsistent dimensions");

		static const int value = mpl::at<0, common>::value;
	};/*}}}*/

	template<int...D> 
	std::size_t common_dim_func(size_t dim)/*{{{*/
	{
		if(common_dim<D...>::value == RUNTIME)
			return dim;
		else
			return common_dim<D...>::value;
	}/*}}}*/
}/*}}}*/

// Result type of operations {{{
template <class T, int M, int N, class U, int O, int P>
struct result_add<Matrix<T,M,N>, Matrix<U,O,P>>/*{{{*/
{
	static_assert(detail::equal_dim<M,O>::value && 
				  detail::equal_dim<N,P>::value,
				  "Mismatched dimensions");

	typedef Matrix<typename result_add_dispatch<T,U>::type,
				   M==RUNTIME ? O : M, 
				   N==RUNTIME ? P : N> type;
};/*}}}*/

template <class T, int M, int N, class U, int O, int P>
struct result_sub<Matrix<T,M,N>, Matrix<U,O,P>>/*{{{*/
{
	static_assert(detail::equal_dim<M,O>::value && 
				  detail::equal_dim<N,P>::value,
				  "Mismatched dimensions");

	typedef Matrix<typename result_sub_dispatch<T,U>::type,
				   M==RUNTIME ? O : M, 
				   N==RUNTIME ? P : N> type;
};/*}}}*/

template <class T, int M, int N, class U, int O, int P>
struct result_mul<Matrix<T,M,N>, Matrix<U,O,P>>/*{{{*/
{
	static_assert(detail::equal_dim<N,O>::value, "Mismatched dimensions");

	typedef Matrix<typename result_mul_dispatch<T,U>::type, M, P> type;
};/*}}}*/

template <class T, int M, int N, class U, int O, int P>
struct result_div<Matrix<T,M,N>, Matrix<U,O,P>>/*{{{*/
{
	static_assert(detail::equal_dim<M,N,O,P>::value, "Matrices must be square");

	typedef Matrix<typename result_div_dispatch<T,U>::type,
				   detail::common_dim<M,N,O,P>::value,
				   detail::common_dim<M,N,O,P>::value> type;
};/*}}}*/

template <class T, int M, int N, class V>
struct result_mul<Matrix<T,M,N>, vector_like<V>>/*{{{*/
{
	static_assert(detail::equal_dim<N,V::dim>::value, "Mismatched dimensions");

	typedef typename result_mul_dispatch<T,typename value_type<V>::type>::type
		subtype;

	typedef typename V::template rebind<subtype,M>::type type;
};/*}}}*/
template <class T, int M, int N, class P>
struct result_mul<Matrix<T,M,N>, point_like<P>>/*{{{*/
{
	static_assert(detail::equal_dim<N,P::dim>::value, "Mismatched dimensions");

	typedef typename result_mul_dispatch<T,typename value_type<P>::type>::type
		subtype;

	typedef typename P::template rebind<subtype,M>::type type;
};/*}}}*/
//}}}

// Matrix

template <class T, int M, int N>
bool is_square(const Matrix<T,M,N> &m)/*{{{*/
{
	// N or M can be RUNTIME...
	return m.rows() == m.cols();
}/*}}}*/

// constructors

template <class T, int M, int N> template <class DUMMY, class>
Matrix<T,M,N>::Matrix(T v)/*{{{*/
{
	static_assert(M==N, "Matrix must be square");

	std::fill(begin(), end(), Vector<T,N>(0));

	for(std::size_t i=0; i<rows(); ++i)
		operator[](i)[i] = v;
}/*}}}*/

template <class T, int M, int N> template <int D>
Matrix<T,M,N>::Matrix(const multi_param<size_t,D,dimension_tag> &d, T v)/*{{{*/
	: coords_base(d)
{
	static_assert(detail::equal_dim<M,N>::value, "Matrix must be square");

	if(!is_square(*this))
		throw std::runtime_error("Matrix must be square");

	Vector<T,M> zero(dim(rows()), 0);
	std::fill(begin(), end(), zero);

	for(std::size_t i=0; i<rows(); ++i)
		operator[](i)[i] = v;
}/*}}}*/

template <class T, int M, int N> 
Matrix<T,M,N>::Matrix(const std::initializer_list<Vector<T,N>> &rows)/*{{{*/
	: coords_base(dim(rows.size(), 
				      N!=RUNTIME ? N 
								 : (rows.size()==0 ? 0 : rows.begin()->size())))
{
	std::copy(rows.begin(), rows.end(), begin());
}/*}}}*/

template <class T, int M, int N> template <class M2>
auto Matrix<T,M,N>::operator=(const M2 &that)/*{{{*/
	-> typename std::enable_if<is_matrix<M2>::value, Matrix &>::type
{
	static_assert(detail::equaldim<Matrix, M2>::value, 
				  "Mismatched dimensions");

	// Assignment works if this is an empty matrix (needed for swapping rows)
	if(!this->empty() && (rows() != that.rows() || cols() != that.cols()))
		throw std::runtime_error("Mismatched dimensions");

	coords_base::operator=(that);
	return *this;
}/*}}}*/

template <class T, int M, int N> template <int O, int P>
auto Matrix<T,M,N>::operator=(Matrix<T,O,P> &&that) -> Matrix &/*{{{*/
{
	static_assert(detail::equaldim<Matrix, Matrix<T,O,P>>::value, 
				  "Mismatched dimensions");

	// Assignment works if this is an empty matrix (needed for swapping rows)
	if(!this->empty() && (rows() != that.rows() || cols() != that.cols()))
		throw std::runtime_error("Mismatched dimensions");

	coords_base::operator=(std::move(that));
	return *this;
}/*}}}*/

// operators

template <class M>
auto operator+=(M &&m, typename value_type<M,2>::type v)/*{{{*/
	-> typename std::enable_if<is_matrix<M>::value,  M&&>::type
{
	static_assert(detail::equal_dim<M::dim_rows,M::dim_cols>::value, 
				  "Matrix must be square");

	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");

	for(std::size_t i=0; i<m.rows(); ++i)
		m[i][i] += v;
	return std::forward<M>(m);
}/*}}}*/

template <class M>
auto operator-=(M &&m, typename value_type<M,2>::type v)/*{{{*/
	-> typename std::enable_if<is_matrix<M>::value, M&&>::typr
{
	static_assert(detail::equal_dim<M::dim_rows,M::dim_cols>::value, 
				  "Matrix must be square");

	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");

	for(std::size_t i=0; i<m.rows(); ++i)
		m[i][i] -= v;
	return std::forward<M>(m);
}/*}}}*/

template <class M>
auto operator*=(M &&m, typename value_type<M,2>::type v)/*{{{*/
	-> typename std::enable_if<is_matrix<M>::value, M&&>::type
{
	for(auto it=m.begin(); it!=m.end(); ++it)
		*it *= v;
	return std::forward<M>(m);
}/*}}}*/

template <class M>
auto operator/=(M &&m, typename value_type<M,2>::type v)/*{{{*/
	-> typename std::enable_if<is_matrix<M>::value, M&&>::type
{
	for(auto it=m.begin(); it!=m.end(); ++it)
		*it /= v;
	return std::forward<M>(m);
}/*}}}*/

template <class M1, class M2>
auto operator+=(M1 &&m1, const M2 &m2)/*{{{*/
	-> typename std::enable_if<is_matrix<M1>::value && 
							   is_matrix<M2>::value, M1&&>::type
{
	static_assert(detail::equaldim<M1, M2>::value,
				"Incompatible matrix dimension");

	if(m1.size() != m2.size())
		throw std::runtime_error("Incompatible matrix dimension");

	auto it2 = m2.begin();
	for(auto it1 = m1.begin(); it1!=m1.end(); ++it1, ++it2)
		*it1 += *it2;

	return std::forward<M1>(m1);
}/*}}}*/

template <class M1, class M2>
auto operator-=(M1 &&m1, const M2 &m2)/*{{{*/
	-> typename std::enable_if<is_matrix<M1>::value && 
							   is_matrix<M2>::value, M1&&>::type
{
	static_assert(detail::equaldim<M1, M2>::value,
				"Incompatible matrix dimension");

	if(m1.size() != m2.size())
		throw std::runtime_error("Incompatible matrix dimension");

	auto it2 = m2.begin();
	for(auto it1 = m1.begin(); it1!=m1.end(); ++it1, ++it2)
		*it1 -= *it2;

	return std::forward<M1>(m1);
}/*}}}*/

template <class M1, class M2>
auto operator*=(M1 &&m1, const M2 &m2) /*{{{*/
	-> typename std::enable_if<is_matrix<M1>::value && 
							   is_matrix<M2>::value, M1&&>::type
{
	return std::forward<M1>(m1 = m1*m2);
}/*}}}*/

template <class T, int M, int N, class U, int O, int P>
auto operator*(const Matrix<T,M,N> &lhs, const Matrix<U,O,P> &rhs)/*{{{*/
	-> typename result_mul_dispatch<Matrix<T,M,N>,Matrix<U,O,P>>::type
{
	if(lhs.cols() != rhs.rows())
		throw std::runtime_error("Incompatible matrix dimensions");

	typename result_mul_dispatch<Matrix<T,M,N>,Matrix<U,O,P>>::type 
		m(dim(lhs.rows(), rhs.cols()));

#if 0
	auto itLr = lhs.begin();
	for(auto itMr = m.begin(); itMr != m.end(); ++itMr, ++itLr)
	{
		int j=0;
		for(auto itMc = itMr->begin(); itMc != itMr->end(); ++itMc, ++j)
		{
			auto itRr = rhs.begin();

			*itMc = *itLr->begin() * *std::next(itRr++->begin(),j);
			for(auto itLc = itLr->begin(); itLc!=itLr->end(); ++itLc, ++itRr)
				*itMc += *itLc * *std::next(itRr->begin(),j);
		}
	}
#endif
#if 1
	for(std::size_t i=0; i<m.rows(); ++i)
	{
		for(std::size_t j=0; j<m.cols(); ++j)
		{
			m[i][j] = lhs[i][0]*rhs[0][j];
			for(std::size_t k=1; k<lhs.cols(); ++k)
				m[i][j] += lhs[i][k]*rhs[k][j];
		}
	}
#endif

	return m;
}/*}}}*/

template <class M1, class M2>
auto operator/=(M1 &&n, const M2 &d)/*{{{*/
	-> typename std::enable_if<is_matrix<M1>::value && 
							   is_matrix<M2>::value, M1&&>::type
{
	return n *= inv(d);
}/*}}}*/

template <class M1, class M2>
auto operator==(const M1 &lhs, const M2 &rhs)/*{{{*/
	-> typename std::enable_if<
			is_matrix<M1>::value && 
			is_matrix<M2>::value,bool>::type
{
	if(lhs.size() != rhs.size())
		return false;
	else
		return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}/*}}}*/


template <class T, int M, int N, class V>
auto operator*(const Matrix<T,M,N> &m, const V &v)/*{{{*/
	-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value),
		typename result_mul_dispatch<Matrix<T,M,N>,V>::type>::type
{
	if(m.cols() != v.size())
		throw std::runtime_error("Mismatched dimensions");

	typename result_mul_dispatch<Matrix<T,M,N>,V>::type ret(dim(v.size()));

	auto itm = m.begin();
	for(auto it=ret.begin(); it!=ret.end(); ++it, ++itm)
		*it = dot(v, *itm);
	return ret;
}/*}}}*/

// identity --------------------------

template <class T, int M> 
Matrix<T,M,M> identity()/*{{{*/
{
	static_assert(M!=RUNTIME, "You must specify the matrix dimension");

	Matrix<T,M,M> identity(1);
	return identity;
}/*}}}*/

template <class T=real> 
const Matrix<T> &identity(std::size_t _d)/*{{{*/
{
	auto d = dim(_d);

	static std::unordered_map<dimension<1>::type, Matrix<T>> identities;

	auto it = identities.find(d);
	if(it == identities.end())
		it = identities.insert({d, Matrix<T>(d,1)}).first;

	return it->second;
}/*}}}*/

template <class T, int M> 
auto identity(std::size_t d)/*{{{*/
	-> typename std::enable_if<M==RUNTIME, const Matrix<T,M,M> &>::type
{
	return identity<T>(d);
}/*}}}*/

template <class T, int M> 
auto identity(std::size_t d)/*{{{*/
	-> typename std::enable_if<M!=RUNTIME, Matrix<T,M,M>>::type
{
	assert(d == M);
	return identity<T,M>();
}/*}}}*/

template <class T, int M, int N> 
bool is_identity(const Matrix<T,M,N> &m)/*{{{*/
{
	static_assert(M==RUNTIME || N==RUNTIME || M==N,
				  "Matrix must be square");

	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");

	return m == identity<T,M>(m.rows());
}/*}}}*/

// zero --------------------------

template <class T, int M, int N, class = 
	typename std::enable_if<M!=RUNTIME && N!=RUNTIME>::type>
Matrix<T,M,N> zero()/*{{{*/
{
	Matrix<T,M,N> zero(0);
	return zero;
}/*}}}*/

template <class T=real> 
const Matrix<T> &zero(size_t rows, size_t cols)/*{{{*/
{
	static std::unordered_map<dimension<2>::type, Matrix<T>> matrices;

	auto d = dim(rows, cols);

	auto it = matrices.find(d);
	if(it == matrices.end())
		it = matrices.insert({d, Matrix<T>(d, 0)}).first;

	return it->second;
}/*}}}*/

template <class T=real, int M> 
auto zero(size_t d)/*{{{*/
	-> typename std::enable_if<M==RUNTIME, const Matrix<T,M,M> &>::type
{
	return zero<T>(d);
}/*}}}*/

template <class T, int M, int N> 
auto zero(size_t d)/*{{{*/
	-> typename std::enable_if<M==RUNTIME || N==RUNTIME, const Matrix<T,M,N> &>::type
{
	static std::unordered_map<dimension<1>::type, Matrix<T,M,N>> matrices;

	auto it = matrices.find(dim(d));
	if(it == matrices.end())
	{
		if(M==RUNTIME && N==RUNTIME)
			it = matrices.insert({dim(d,d), Matrix<T,M,N>(dim(d), 0)}).first;
		else if(M==RUNTIME)
			it = matrices.insert({dim(d,N), Matrix<T,M,N>(dim(d,N), 0)}).first;
		else
		{
			assert(N==RUNTIME);
			it = matrices.insert({dim(M,d), Matrix<T,M,N>(dim(M,d), 0)}).first;
		}
	}

	return it->second;
}/*}}}*/

template <class T, int M, int N, class = 
	typename std::enable_if<M==RUNTIME && N==RUNTIME>::type>
const Matrix<T> &zero(size_t rows, size_t cols)/*{{{*/
{
	return zero<T>(rows,cols);
}/*}}}*/

template <class T, int M> 
auto zero(size_t d)/*{{{*/
	-> typename std::enable_if<M!=RUNTIME, const Matrix<T,M,M> &>::type
{
	assert(d == M);
	return zero<M,T>();
}/*}}}*/

template <class T, int M, int N> 
bool is_zero(const Matrix<T,M,N> &m)/*{{{*/
{
	static_assert(M==RUNTIME || N==RUNTIME || M==N,
				  "Matrix must be square");

	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");

	return m == zero<M,T>(m.rows());
}/*}}}*/

template <class T, int N>
Matrix<T,N,N> diag(const Vector<T,N> &v)/*{{{*/
{
	Matrix<T,N,N> d(dim(v.size(), v.size()), 0);

	for(std::size_t i=0; i<d.rows(); ++i)
		d[i][i] = v[i];

	return d;
}/*}}}*/

// SVD decomposition -------------------------

template <class T, int M, int N>
bool svd_inplace(Matrix<T,M,N> &a, Vector<T,N> &w, Matrix<T,N,N> &v)/*{{{*/
{
	Matrix<T,M,N> &u = a;

	// Decomposition

	Vector<T,N> rv1(dim(a.cols()));

	static const T eps = std::numeric_limits<T>::epsilon();

	T g = 0,
	  scale = 0,
	  anorm = 0;

	// Householder reduction to bidiagonal form
	for(size_t i=0; i<a.cols(); ++i) 
	{
		size_t l = i+2;
		rv1[i] = scale*g;

		g = 0;
		scale = 0;

		T s = 0;

		if(i < a.rows()) 
		{
			for(size_t k=i; k<a.rows(); ++k) 
				scale += abs(u[k][i]);

			if(scale != 0) 
			{
				for(size_t k=i; k<a.rows(); ++k) 
				{
					u[k][i] /= scale;
					s += u[k][i]*u[k][i];
				}
				T f = u[i][i];
				g = -copysign(sqrt(s),f);
				T h = f*g-s;
				u[i][i] = f-g;
				for(size_t j=l-1; j<a.cols(); ++j) 
				{
					s = 0;
					for(size_t k=i; k<a.rows(); ++k) 
						s += u[k][i]*u[k][j];

					f=s/h;

					for(size_t k=i; k<a.rows(); ++k) 
						u[k][j] += f*u[k][i];
				}
				for(size_t k=i; k<a.rows(); ++k) 
					u[k][i] *= scale;
			}
		}

		w[i] = scale*g;
		g = s = scale = 0.0;
		if(i+1 <= a.rows() && i+1 != a.cols()) 
		{
			for(size_t k=l-1; k<a.cols(); ++k) 
				scale += abs(u[i][k]);
			if(scale != 0.0) 
			{
				for(size_t k=l-1; k<a.cols(); ++k) 
				{
					u[i][k] /= scale;
					s += u[i][k]*u[i][k];
				}
				T f = u[i][l-1];
				g = -copysign(sqrt(s),f);
				T h = f*g-s;
				u[i][l-1] = f-g;

				for(size_t k=l-1; k<a.cols(); ++k) 
					rv1[k] = u[i][k]/h;
				for(size_t j=l-1; j<a.rows(); ++j) 
				{
					s = 0;
					for(size_t k=l-1; k<a.cols(); ++k) 
						s += u[j][k]*u[i][k];
					for(size_t k=l-1; k<a.cols(); ++k) 
						u[j][k] += s*rv1[k];
				}
				for(size_t k=l-1; k<a.cols(); ++k) 
					u[i][k] *= scale;
			}
		}
		anorm = max(anorm, abs(w[i]) + abs(rv1[i]));
	}

	// Accumulation of right-hand transformations
	for(size_t i=a.cols()-1; (int)i>=0; --i) 
	{
		size_t l;
		if(i < a.cols()-1) 
		{
			if(g != 0) 
			{
				for(size_t j=l; j<a.cols(); ++j)
				{
					// double divition to avoid possible underflow
					v[j][i] = (u[i][j]/u[i][l])/g;
				}
				for(size_t j=l; j<a.cols(); ++j) 
				{
					T s = 0;
					for(size_t k=l; k<a.cols(); ++k) 
						s += u[i][k]*v[k][j];

					for(size_t k=l; k<a.cols(); ++k) 
						v[k][j] += s*v[k][i];
				}
			}
			for(size_t j=l; j<a.cols(); ++j) 
				v[i][j] = v[j][i] = 0;
		}
		v[i][i] = 1;
		g = rv1[i];
		l = i;
	}

	// Accumulation of left-hand transformations
	for(size_t i=min(a.rows(),a.cols())-1; (int)i>=0; i--)
	{
		size_t l = i+1;
		g = w[i];
		for(size_t j=l; j<a.cols(); ++j) 
			u[i][j]=0.0;
		if(g != 0.0) 
		{
			g = 1.0/g;
			for(size_t j=l; j<a.cols(); ++j) 
			{
				T s = 0;
				for(size_t k=l; k<a.rows(); ++k) 
					s += u[k][i]*u[k][j];
				T f = (s/u[i][i])*g;
				for(size_t k=i; k<a.rows(); ++k) 
					u[k][j] += f*u[k][i];
			}
			for(size_t j=i; j<a.rows(); ++j) 
				u[j][i] *= g;
		} 
		else 
		{
			for(size_t j=i; j<a.rows(); ++j) 
				u[j][i] = 0;
		}
		++u[i][i];
	}

	// Diagonalizatoin of the bidiagonal form.
	// Loops over singular values
	for(size_t k=a.cols()-1; (int)k>=0; k--) 
	{
		// Loops over allowed iterations
		for(size_t its=0; its<30; its++) 
		{
			bool flag = true;
			size_t l, nm;

			// Test for splitting
			for(l=k; (int)l>=0; l--) 
			{
				nm = l-1;
				// Note that rv1[0] is always zero
				if(l == 0 || abs(rv1[l]) <= eps*anorm) 
				{
					flag = false;
					break;
				}
				if(abs(w[nm]) <= eps*anorm) 
					break;
			}

			if(flag) 
			{
				// Cancellation of rv1[l], if l > 0
				T c = 0,
				  s = 1;
				for(size_t i=l; i<k+1; i++) 
				{
					T f = s*rv1[i];
					rv1[i] = c*rv1[i];
					if(abs(f) <= eps*anorm) 
						break;
					g = w[i];
					T h = hypot(f,g);
					w[i] = h;
					h = 1.0/h;
					c = g*h;
					s = -f*h;
					for(size_t j=0; j<a.rows(); ++j) 
					{
						T y = u[j][nm];
						T z = u[j][i];
						u[j][nm] = y*c + z*s;
						u[j][i] = z*c - y*s;
					}
				}
			}
			T z = w[k];
			if(l == k) // Convergence
			{
				if(z < 0.0) // Singular value is made nonnegative
				{
					w[k] = -z;
					for(size_t j=0; j<a.cols(); ++j) 
						v[j][k] = -v[j][k];
				}
				break;
			}
			if(its == 29)
				return false;

			// Shift from bottom 2-by-2 minor
			T x = w[l];
			nm = k-1;
			T y = w[nm];
			g = rv1[nm];
			T h = rv1[k],
			  f = ((y-z)*(y+z) + (g-h)*(g+h))/(2*h*y);
			g = hypot(f,1);
			f = ((x-z)*(x+z) + h*((y/(f+copysign(g,f)))-h))/x;
			T s = 1,
			  c = 1;
			// Next QR transformation
			for(size_t j=l; j<=nm; ++j) 
			{
				size_t i = j+1;

				g = rv1[i];
				T y = w[i];
				T h = s*g;
				g = c*g;
				T z = hypot(f,h);
				rv1[j] = z;
				c = f/z;
				s = h/z;
				f = x*c + g*s;
				g = g*c - x*s;
				h = y*s;
				y *= c;
				for(size_t jj=0; jj<a.cols(); ++jj) 
				{
					T x = v[jj][j];
					T z = v[jj][i];
					v[jj][j] = x*c + z*s;
					v[jj][i] = z*c - x*s;
				}
				z = hypot(f,h);
				w[j] = z;
				// Rotation can be arbitrary if z == 0
				if(z) 
				{
					z = 1.0/z;
					c = f*z;
					s = h*z;
				}
				f = c*g + s*y;
				x = c*y - s*g;
				for(size_t jj=0; jj<a.rows(); ++jj) 
				{
					T y = u[jj][j];
					T z = u[jj][i];
					u[jj][j] = y*c + z*s;
					u[jj][i] = z*c - y*s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	
	// Reorder

	Vector<T,M> su(dim(a.rows()));
	Vector<T,N> sv(dim(a.cols()));

	size_t inc=1;
	do 
	{ 
		inc *= 3; 
		++inc; 
	} 
	while(inc <= a.cols());

	do 
	{
		inc /= 3;
		for(size_t i=inc; i<a.cols(); ++i) 
		{
			T sw = w[i];
			for(size_t k=0; k<a.rows(); ++k) 
				su[k] = u[k][i];

			for(size_t k=0; k<a.cols(); ++k) 
				sv[k] = v[k][i];

			size_t j = i;
			while(w[j-inc] < sw) 
			{
				w[j] = w[j-inc];
				for(size_t k=0; k<a.rows(); ++k) 
					u[k][j] = u[k][j-inc];

				for(size_t k=0; k<a.cols(); ++k) 
					v[k][j] = v[k][j-inc];

				assert(j >= inc);
				j -= inc;
				if (j < inc) 
					break;
			}
			w[j] = sw;
			for(size_t k=0; k<a.rows(); ++k) 
				u[k][j] = su[k];

			for(size_t k=0; k<a.cols(); ++k) 
				v[k][j] = sv[k];
		}
	} 
	while(inc > 1);

	for(size_t k=0; k<a.cols(); ++k) 
	{
		size_t s = 0;
		for(size_t i=0; i<a.rows(); ++i) 
		{
			if (u[i][k] < 0) 
				++s;
		}

		for(size_t j=0; j<a.cols(); ++j) 
		{
			if(v[j][k] < 0) 
				s++;
		}

		if(s > (a.rows()+a.cols())/2) 
		{
			for(size_t i=0; i<a.rows(); ++i) 
				u[i][k] = -u[i][k];

			for(size_t j=0; j<a.cols(); ++j) 
				v[j][k] = -v[j][k];
		}
	}
	return true;
}/*}}}*/

template <class T, int M, int N>
auto svd(const Matrix<T,M,N> &m)/*{{{*/
	-> std::tuple<Matrix<T,M,N>, Vector<T,N>, Matrix<T,N,N>>
{
	auto USV = std::make_tuple(m, Vector<T,N>(dim(m.cols())), 
							   Matrix<T,N,N>(dim(m.cols(), m.cols())));

	svd_inplace(get<0>(USV), get<1>(USV), get<2>(USV));

	return USV;
}/*}}}*/

// unit --------------------------

template <class T, int M, int N>
Matrix<T,M,N> &unit_inplace(Matrix<T,M,N> &m)/*{{{*/
{
	T d = det(m); // this will constraint M==N
	if(equal(d,0))
		m = identity<T,M>(m.size());
	else
		m /= d;

	return m;
}/*}}}*/

template <class T, int M, int N>
Matrix<T,M,N> unit(const Matrix<T,M,N> &m)/*{{{*/
{
	Matrix<T,M,N> copy(m);
	unit_inplace(copy);
	return copy;
}/*}}}*/

// trace --------------------------

template <class T, int M, int N> 
T trace(const Matrix<T,M,N> &m)/*{{{*/
{
	static_assert(M==RUNTIME || N==RUNTIME || M==N,
				  "Matrix must be square");

	if(!is_square(m))
		throw std::runtime_error("Matrix must be squared");

	T t = 0;
	for(std::size_t i=0; i<m.rows(); ++i)
		t += m[i][i];
	return t;
}/*}}}*/

// LU decomposition ------------

template <class T, int M, int N, int P=detail::common_dim<M,N>::value> 
void lu_inplace(Matrix<T,M,N> &m, Vector<int,P> *p=NULL, T *d=NULL)/*{{{*/
{
	static_assert(M==RUNTIME || N==RUNTIME || M==N,
				  "Matrix must be square");
	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");

	static_assert(detail::equal_dim<M,N,P>::value, 
				  "Incompatible vector dimension");

	if(p && m.rows() != p->size())
		throw std::runtime_error("Incompatible vector dimension");

	// Ref: Numerical Recipes in C++, 3rd ed, Chapter 2, pp. 52-53

	// Crout's algorithm with implicit pivoting (based on partial pivoting)

	// stores the implicit scaling of each row
	Vector<T,detail::common_dim<M,N,P>::value> vv(dim(m.rows())); 

	// Loop over rows to get the implicit scaling information
	for(std::size_t i=0; i<vv.size(); ++i)
	{
		T big = 0;
		for(std::size_t j=0; j<vv.size(); ++j)
			big = max(big,abs(m[i][j]));
		if(big == 0)
			throw std::runtime_error("Singular matrix in lu_into");
		vv[i] = 1/big;
	}

	if(d)
		*d = 1;	// no row interchanges yet

	for(std::size_t k=0; k<vv.size(); ++k)
	{
		// Initialize for the search for largest pivot element
		T big = 0; 
		std::size_t imax=k;
		for(std::size_t i=k; i<vv.size(); ++i)
		{
			// Is the figure of merit for the pivot better than the 
			// best so far?
			T aux = vv[i]*abs(m[i][k]);
			if(aux > big)
			{
				big = aux;
				imax = i;
			}
		}

		// Do we need to interchange rows?
		if(k != imax)
		{
			// Do it
			swap(m[imax], m[k]);

			if(d)
				*d = -*d;

			vv[imax] = vv[k]; // interchange the scale factor
		}

		if(p)
			(*p)[k] = imax;

		// If the pivot element is zero the matrix is singular (at least to 
		// the precision of the algorithm). For some applications on singular
		// matrices, it is desirable to substitute EPSILON for zero
		if(m[k][k] == 0)
		   m[k][k] = 1e-20;

		// Now, finally, divide by the pivot element
		for(std::size_t i=k+1; i<vv.size(); ++i)
		{
			T aux;
			aux = m[i][k] /= m[k][k];
			for(std::size_t j=k+1; j<vv.size(); ++j)
				m[i][j] -= aux*m[k][j];
		}
	}
}/*}}}*/

template <class T, int M, int N, int P=detail::common_dim<M,N>::value> 
auto lu(const Matrix<T,M,N> &m, Vector<int,P> *p, T *d=NULL)/*{{{*/
	-> Matrix<T,detail::common_dim<M,N,P>::value>
{
	Matrix<T,detail::common_dim<M,N,P>::value> lu = m;
	lu_inplace(lu, p, d);
	return lu;
}/*}}}*/

template <class T, int M, int N> 
auto lu(const Matrix<T,M,N> &m)/*{{{*/
	-> std::tuple<Matrix<T,detail::common_dim<M,N>::value>,
	              Matrix<T,detail::common_dim<M,N>::value>,
	              Matrix<T,detail::common_dim<M,N>::value>>
{
	Matrix<T,detail::common_dim<M,N>::value> lu = m;
	Vector<int,detail::common_dim<M,N>::value> p(dim(lu.rows()));
	lu_inplace(lu, &p);

	auto LUP = 
	   std::make_tuple(Matrix<T,detail::common_dim<M,N>::value>(dim(lu.rows())),
	   			       Matrix<T,detail::common_dim<M,N>::value>(dim(lu.rows())),
	   			       identity<T,detail::common_dim<M,N>::value>(lu.rows()));

	for(std::size_t i=0; i<lu.rows(); ++i)
	{
		swap(get<2>(LUP)[i], get<2>(LUP)[p[i]]);

		std::size_t j;
		for(j=0; j<i; ++j)
		{
			get<0>(LUP)[i][j] = lu[i][j];
			get<1>(LUP)[i][j] = 0;
		}

		get<0>(LUP)[i][j] = 1;

		for(;j<lu.rows(); ++j)
		{
			if(i!=j)
				get<0>(LUP)[i][j] = 0;
			get<1>(LUP)[i][j] = lu[i][j];
		}
	}

	return LUP;
}/*}}}*/

// transpose -----------------

template <class T, int M, int N> 
Matrix<T,M,N> &transpose_inplace(Matrix<T,M,N> &m)/*{{{*/
{
	static_assert(M==RUNTIME || N==RUNTIME || M==N,
				  "Matrix must be square");

	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");

	for(std::size_t i=0; i<m.rows(); ++i)
		for(std::size_t j=i+1; j<m.cols(); ++j)
			swap(m[i][j], m[j][i]);

	return m;
}/*}}}*/

template <class T, int M, int N> 
Matrix<T,N,M> transpose(const Matrix<T,M,N> &m)/*{{{*/
{
	Matrix<T,N,M> r(dim(m.cols(), m.rows()));

	for(unsigned i=0; i<m.rows(); ++i)
		for(unsigned j=0; j<m.cols(); ++j)
			r[j][i] = m[i][j];

	return r;
}/*}}}*/

// determinant -----------------

namespace detail
{
	template <class T, int M, int N> 
	T det(const Matrix<T,M,N> &m, mpl::int_<0>)/*{{{*/
	{
		assert(is_square(m) && m.rows() == 0);

		return 1; // surprising, isn't it?
	}/*}}}*/
	template <class T, int M, int N> 
	T det(const Matrix<T,M,N> &m, mpl::int_<1>)/*{{{*/
	{
		assert(is_square(m) && m.rows() == 1);
		return m[0][0];
	}/*}}}*/
	template <class T, int M, int N> 
	T det(const Matrix<T,M,N> &m, mpl::int_<2>)/*{{{*/
	{
		assert(is_square(m) && m.rows() == 2);

		return m[0][0]*m[1][1] - m[0][1]*m[1][0];
	}/*}}}*/
	template <class T, int M, int N> 
	T det(const Matrix<T,M,N> &m, mpl::int_<3>)/*{{{*/
	{
		assert(is_square(m) && m.rows() == 3);

		return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) +
			   m[0][1]*(m[1][2]*m[2][0] - m[1][0]*m[2][2]) +
			   m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
	}/*}}}*/
	template <class T, int M, int N, int D> 
	T det(const Matrix<T,M,N> &m, mpl::int_<D>)/*{{{*/
	{
		assert(is_square(m) && (D == RUNTIME || (int)m.rows() == D));

		T d;
		auto LU = lu(m, (Vector<int,D> *)NULL, &d);

		for(std::size_t i=0; i<(D==RUNTIME?m.rows():D); ++i)
			d *= LU[i][i];
		return d;
	}/*}}}*/
}

template <class T, int M, int N>
auto det(const Matrix<T,M,N> &m)/*{{{*/
	-> typename std::enable_if<M!=RUNTIME || N!=RUNTIME, T>::type
{
	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");

	return detail::det(m,mpl::int_<M==RUNTIME ? N : M>());
}/*}}}*/

template <class T>
T det(const Matrix<T,RUNTIME,RUNTIME> &m)/*{{{*/
{
	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");

	switch(m.rows())
	{
	case 0:
		return detail::det(m, mpl::int_<0>());
	case 1:
		return detail::det(m, mpl::int_<1>());
	case 2:
		return detail::det(m, mpl::int_<2>());
	case 3:
		return detail::det(m, mpl::int_<3>());
	default:
		return detail::det(m, mpl::int_<RUNTIME>());
	}
}/*}}}*/

// solve -----------------------

template <class T, int M, int N, int O, int P,
		  template<class,int> class V>
auto solve_inplace(const Matrix<T,M,N> &lu, const Vector<int,O> &p, V<T,P> &b)/*{{{*/
	-> typename requires<is_vector<V<T,P>>,void>::type
{
	static_assert(M==RUNTIME || N==RUNTIME || M==N,
				  "Matrix must be square");
	if(!is_square(lu))
		throw std::runtime_error("Matrix must be square");

	static_assert(detail::equal_dim<M,N,O>::value,
				  "Incompatible vector dimension");

	static_assert(detail::equal_dim<M,N,P>::value,
				  "Incompatible vector dimension");

	if(lu.rows() != p.size())
		throw std::runtime_error("Incompatible vector dimension");

	if(lu.rows() != b.size())
		throw std::runtime_error("Incompatible vector dimension");

	// Ref: Numerical Recipes in C, 2nd ed, Chapter 2.3, p. 47
	
	// We now do the forward substitution.
	int ii=-1;
	for(std::size_t i=0; i<detail::common_dim_func<M,N,O,P>(lu.rows()); ++i)
	{
		int ip = p[i];
		T sum = b[ip];
		b[ip] = b[i];

		// When ii>=0, it will become the index of the first 
		// nonvanishing element of b. 
		if(ii>=0)
		{
			for(std::size_t j=ii; j<i; ++j)
				sum -= lu[i][j]*b[j];
		}
		else if(sum != 0)
			ii = i;

		b[i] = sum;
	}

	// Now to the back substitution
	for(std::size_t i=detail::common_dim_func<M,N,O,P>(lu.rows())-1; (int)i>=0; --i)
	{
		T sum = b[i];
		for(std::size_t j=i+1; j<detail::common_dim_func<M,N,O,P>(lu.rows()); ++j)
			sum -= lu[i][j]*b[j];
		b[i] = sum/lu[i][i];
	}
}/*}}}*/
template <class T, int M, int N, int O> 
void solve_inplace(const Matrix<T,M,N> &m, Vector<T,O> &b)/*{{{*/
{
	static_assert(detail::equal_dim<M,N>::value,
				  "Matrix must be square");

	static_assert(detail::equal_dim<M,N,O>::value,
				  "Inconsistent vector dimension");

	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");
	if(m.rows() != b.size())
		throw std::runtime_error("Inconsistent vector dimension");

	Vector<int,detail::common_dim<M,N,O>::value> p(dim(m.rows()));
	auto lu = lu(m, &p);
	solve_inplace(lu, p, b);
}/*}}}*/
template <class T, int M, int N, int O, int P> 
Vector<T,M> solve(const Matrix<T,M,N> &lu, const Vector<int,O> &p, const Vector<T,P> &b)/*{{{*/
{
	static_assert(M==RUNTIME || N==RUNTIME || M==N,
				  "Matrix must be square");
	if(!is_square(lu))
		throw std::runtime_error("Matrix must be square");

	static_assert(detail::equal_dim<M,N,O>::value,
				  "Incompatible vector dimension");

	static_assert(detail::equal_dim<M,N,P>::value,
				  "Incompatible vector dimension");

	if(lu.rows() != p.size())
		throw std::runtime_error("Incompatible vector dimension");

	if(lu.rows() != b.size())
		throw std::runtime_error("Incompatible vector dimension");

	auto x = b;
	solve_inplace(lu, p, x);
	return x;
}/*}}}*/
template <class T, int M, int N, int O> 
Vector<T,M> solve(const Matrix<T,M,N> &m, const Vector<T,O> &b)/*{{{*/
{
	static_assert(detail::equal_dim<M,N>::value,
				  "Matrix must be square");

	static_assert(detail::equal_dim<M,N,O>::value,
				  "Inconsistent vector dimension");

	if(!is_square(m))
		throw std::runtime_error("Matrix must be square");
	if(m.rows() != b.size())
		throw std::runtime_error("Inconsistent vector dimension");

	Vector<int,detail::common_dim<M,N,O>::value> p(dim(m.rows()));
	auto LU = lu(m, &p);
	return solve(LU, p, b);
}/*}}}*/

// factors ---------------------

template <class T, int M> 
Matrix<T,M-1,M-1> minor(const Matrix<T,M,M> &m, int ci, int cj)/*{{{*/
{
	Matrix<T,M-1,M-1> r;
	for(int i=0, ii=0; i<M; ++i)
	{
		if(i == ci)
			continue;
		for(int j=0, jj=0; j<M; ++j)
		{
			if(j==cj)
				continue;

			r[ii][jj++] = m[i][j];
		}

		++ii;
	}
	return det(r);
}/*}}}*/
template <class T, int M> 
T cofactor(const Matrix<T,M,M> &m, int i, int j)/*{{{*/
{
	T min = minor(m, i, j);

	if((i+j) & 1)
		return min;
	else
		return -min;
}/*}}}*/
template <class T, int M> 
Matrix<T,M,M> cofactor(const Matrix<T,M,M> &m)/*{{{*/
{
	Matrix<T,M,M> r;
	for(int i=0; i<M; ++i)
		for(int j=0; j<M; ++j)
			r[i][j] = cofactor(m, i, j);
	return r;
}/*}}}*/
template <class T, int M> 
Matrix<T,M,M> adjugate(const Matrix<T,M,M> &m)/*{{{*/
{
	return transpose(cofactor_matrix(m));
}/*}}}*/

// inverse -----------------------

template <class T, int M, int N, int O, int P> 
void inv_lu(Matrix<T,M,N> &out, Matrix<T,O,P> &m)/*{{{*/
{
	static_assert(detail::equal_dim<M,N,O,P>::value,
				  "Incompatible matrix dimension");

	// Ref: Numerical Recipes in C, 2nd ed, Chapter 2.3, p. 48
	
	Vector<int,detail::common_dim<M,N,O,P>::value> p(dim(out.rows()));
	lu_inplace(m, &p);

	out = identity<T,detail::common_dim<M,N,O,P>::value>(p.size());

	for(auto it=out.begin(); it!=out.end(); ++it)
		solve_inplace(m, p, *it);

	transpose_inplace(out);
}/*}}}*/

namespace detail
{
	template <class T, int M, int N> 
	void inv_inplace(Matrix<T,M,N> &m, mpl::int_<1>)/*{{{*/
	{
		assert(m.rows() == 1);

		m[0][0] = 1/m[0][0];
	}/*}}}*/
	template <class T, int M, int N> 
	void inv_inplace(Matrix<T,M,N> &m, mpl::int_<2>)/*{{{*/
	{
		assert(m.rows() == 2);

		T d = det(m);

		m[0][0] =  m[1][1]/d;
		m[0][1] = -m[0][1]/d;
		m[1][0] = -m[1][0]/d;
		m[1][1] =  m[0][0]/d;
	}/*}}}*/
	template <class T, int M, int N, int D> 
	void inv_inplace(Matrix<T,M,N> &m, mpl::int_<D>)/*{{{*/
	{
		Matrix<T,M,N> y(dim(m.rows()));
		inv_lu(y, m);
		m = std::move(y);
	}/*}}}*/
}

template <class T, int M, int N>
Matrix<T,M,N> &inv_inplace(Matrix<T,M,N> &m)/*{{{*/
{
	static_assert(M==RUNTIME || N==RUNTIME || M==N,
				  "Matrix must be square");

	if(m.rows() != m.cols())
		throw std::runtime_error("Matrix must be square");

	switch(m.rows())
	{
	case 1:
		detail::inv_inplace(m, mpl::int_<1>());
		break;
	case 2:
		detail::inv_inplace(m, mpl::int_<2>());
		break;
	default:
		detail::inv_inplace(m, mpl::int_<RUNTIME>());
		break;
	}

	return m;
}/*}}}*/

template <class T, int M>
auto inv_inplace(Matrix<T,M> &m)/*{{{*/
	-> typename std::enable_if<M!=RUNTIME,Matrix<T,M> &>::type
{
	detail::inv_inplace(m, mpl::int_<M>());
	return m;
}/*}}}*/

template <class T, int M, int N>
Matrix<T,M,N> inv(Matrix<T,M,N> m)/*{{{*/
{
	inv_inplace(m);
	return m;
}/*}}}*/

template <class T, int M, int N> 
std::ostream &operator<<(std::ostream &out, const Matrix<T,M,N> &m)/*{{{*/
{
	for(std::size_t i=0; i<m.rows(); ++i)
	{
		out << m[i];
		if(i < m.rows()-1)
			out << ";";
	}
	return out;
}/*}}}*/

// submatrix -------------------------

template <int...ROWS> struct rows/*{{{*/
{
	typedef mpl::vector_c<int,ROWS...> type;
};/*}}}*/
template <int...COLS> struct cols/*{{{*/
{
	typedef mpl::vector_c<int,COLS...> type;
};/*}}}*/
template <int...ROWS> struct subrows/*{{{*/
{
	typedef mpl::vector_c<int,ROWS...> type;
};/*}}}*/
template <int...COLS> struct subcols/*{{{*/
{
	typedef mpl::vector_c<int,COLS...> type;
};/*}}}*/

namespace detail
{
	template <template<int...>class KIND, class...ARGS> 
	struct get_items : get_items<KIND, typename unref<ARGS>::type...> {};/*{{{*/
	template <template<int...> class KIND> struct get_items<KIND>
	{
		typedef mpl::vector_c<int> type;
	};

	// Jumps over things we do not want
	template <template<int...> class KIND, class...ARGS, 
			 template <int...> class JUNK, int...COLS> 
	struct get_items<KIND, JUNK<COLS...>,ARGS...>
	{
		typedef typename get_items<KIND, ARGS...>::type type;
	};

	template <template<int...> class KIND, class...ARGS, int...ROWS> 
	struct get_items<KIND, KIND<ROWS...>,ARGS...>
	{
		typedef typename mpl::unique 
		<
			mpl::add<mpl::vector_c<int,ROWS...>, get_items<KIND,ARGS...>>
		>::type type;
	};/*}}}*/

	template <int M, class...ITEMS> struct get_rows/*{{{*/
	{
		typedef typename get_items<rows,ITEMS...>::type row_items;
		typedef typename get_items<subrows,ITEMS...>::type subrow_items;

		typedef typename std::conditional
		<
			(row_items::size > 0),
			mpl::sub<row_items, subrow_items>,
			mpl::sub<mpl::create_range<int, 0,M>, subrow_items>
		>::type::type type;
	};/*}}}*/

	template <int M, class...ITEMS> struct get_cols/*{{{*/
	{
		typedef typename get_items<cols,ITEMS...>::type col_items;
		typedef typename get_items<subcols,ITEMS...>::type subcol_items;

		typedef typename std::conditional
		<
			(col_items::size > 0),
			mpl::sub<col_items, subcol_items>,
			mpl::sub<mpl::create_range<int,0,M>, subcol_items>
		>::type::type type;
	};/*}}}*/

	template <int DI,int DJ, int SI,int SJ, int M,int N, class ROWS, class COLS>
	struct process_cols/*{{{*/
	{
		typedef typename std::conditional
		<
			DJ < COLS::size,
			typename std::conditional
			<
				mpl::exists_c<COLS,SJ>::value, 
				std::identity<process_cols<DI,DJ,SI,SJ,M,N,ROWS,COLS>>,
				process_cols<DI,DJ,SI,SJ+1,M,N,ROWS,COLS>
			>::type,
			std::identity<null_functor>
		>::type::type type;

		template <class T>
		process_cols(Matrix<T,ROWS::size,COLS::size> &ret, 
					 const Matrix<T,M,N> &m)
		{
			static_assert(DI<ROWS::size && SI<M, "Row out of bounds");
			static_assert(DJ<COLS::size && SJ<N, "Column out of bounds");

			ret[DI][DJ] = m[SI][SJ];

			typename process_cols<DI,DJ+1,SI,SJ+1,M,N,ROWS,COLS>::type(ret, m);
		}
	};/*}}}*/

	template <int DI, int SI, int M, int N, class ROWS, class COLS>
	struct process_rows/*{{{*/
	{
		typedef typename std::conditional
		<
			DI < ROWS::size,
			typename std::conditional
			<
				mpl::exists_c<ROWS,SI>::value, 
				std::identity<process_rows<DI,SI,M,N,ROWS,COLS>>,
				process_rows<DI,SI+1,M,N,ROWS,COLS>
			>::type,
			std::identity<null_functor>
		>::type::type type;

		template <class T>
		process_rows(Matrix<T,ROWS::size,COLS::size> &ret, 
					 const Matrix<T,M,N> &m)
		{
			typename process_cols<DI,0,SI,0,M,N,ROWS,COLS>::type(ret, m);
			typename process_rows<DI+1,SI+1,M,N,ROWS,COLS>::type(ret, m);
		}
	};/*}}}*/
}

template <class... ITEMS,
		  class T, int M, int N>
auto submatrix(const Matrix<T,M,N> &m)/*{{{*/
	-> Matrix<T,detail::get_rows<M,ITEMS...>::type::size, 
			    detail::get_cols<N,ITEMS...>::type::size> 
{
	typedef typename detail::get_rows<M,ITEMS...>::type ROWS;
	typedef typename detail::get_cols<N,ITEMS...>::type COLS;

	Matrix<T, ROWS::size, COLS::size> ret;

	typename detail::process_rows<0,0,M,N,ROWS,COLS>::type(ret, m);

	return ret;
}/*}}}*/

template <int O, int P, class T, int M, int N>
Matrix<T,O,P> identity_augment(const Matrix<T,M,N> &m)/*{{{*/
{
	Matrix<T,O,P> out;

	for(unsigned i=0; i<O; ++i)
	{
		if(i < M)
		{
			for(unsigned j=0; j<P; ++j)
			{
				if(j < N)
					out[i][j] = m[i][j];
				else if(i==j)
					out[i][j] = 1;
				else
					out[i][j] = 0;
			}
		}
		else
		{
			for(unsigned j=0; j<P; ++j)
			{
				if(i==j)
					out[i][j] = 1;
				else
					out[i][j] = 0;
			}
		}
	}
	return out;
}/*}}}*/

}} // namespace s3d::math

// $Id: matrix.hpp 3086 2010-08-31 18:46:04Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

