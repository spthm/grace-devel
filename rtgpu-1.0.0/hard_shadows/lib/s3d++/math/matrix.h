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

#ifndef S3D_MATH_MATRIX_H
#define S3D_MATH_MATRIX_H

#include <array>
#include <vector>
#include <initializer_list>
#include "../mpl/at.h"
#include "matrix_space.h"
#include "r2/size.h"
#include "fwd.h"
#include "vector_view.h"
#include "real.h"

// I hate macros
#ifdef minor
#   undef minor
#endif

namespace s3d { namespace math
{

template <class T>
struct is_matrix
{
	static const bool value = false;
};

template <class T>
struct is_matrix<T&> : is_matrix<typename std::remove_cv<T>::type>
{
};

template <class T, int M, int N> class Matrix
	: public coords<Matrix<T,M,N>,matrix_space<T,M,N>>
	
{
private:
	typedef coords<Matrix,matrix_space<T,M,N>> coords_base;
public:
	using coords_base::value_type;

	static const size_t 
		dim_rows = mpl::at<0,typename traits::dim<Matrix>::type>::value,
		dim_cols = mpl::at<1,typename traits::dim<Matrix>::type>::value;

	template <class U, int O=M, int P=N> 
	struct rebind { typedef Matrix<U,M,N> type; };

	Matrix() {} 

	explicit Matrix(const dimension<2>::type &d) 
		: coords_base(d) {}

	template <class U, int O, int P>
	Matrix(Matrix<U,O,P> &&d) 
		: coords_base(std::move(d)) {}

	// May specify one dimension if the other is static
	template <class DUMMY=int>
	explicit Matrix(const dimension<1>::type &d,
		typename std::enable_if<sizeof(DUMMY) &&
			((M==RUNTIME) ^ (N==RUNTIME))>::type* = NULL)
		: coords_base(d) {}

	// (static) square matrix? Just one dimension suffices
	template <class DUMMY=int>
	explicit Matrix(const dimension<1>::type &d,
		typename std::enable_if<sizeof(DUMMY) &&
			M==N>::type* = NULL)
		: coords_base(dim(d[0], d[0])) {}

	template <class U, int O, int P> 
	Matrix(const Matrix<U,O,P> &m) : coords_base(m) {}

	template <class DUMMY=int, class = typename std::enable_if<sizeof(DUMMY) && 
			M!=RUNTIME && N!=RUNTIME>::type>
	Matrix(T v);

	template <int D> // D==1 or D==2
	Matrix(const multi_param<size_t,D,dimension_tag> &d, T v);

	Matrix(const std::initializer_list<Vector<T,N>> &rows);

	size_t rows() const 
		{ return std::distance(begin(), end()); }

	size_t cols() const 
		{ return N!=RUNTIME ? N : (rows()==0 ? 0 : begin()->size()); }

	r2::usize size() const { return {rows(), cols()}; }

	template<class M2>
	auto operator=(const M2 &that)
		-> typename std::enable_if<is_matrix<M2>::value,Matrix &>::type;

	template<int O, int P>
	Matrix &operator=(Matrix<T,O,P> &&that);

	using coords_base::begin;
	using coords_base::end;
	using coords_base::operator[];
};

template <class T, int M, int N>
struct is_matrix<Matrix<T,M,N>>
{
	static const bool value = true;
};

template <class T>
struct matrix_like : concept<T> {};

template <class T, int M, int N>
struct is_view<matrix_space_view<T,M,N>>
{
	static const bool value = true;
};

} // namespace math

template <class T, int M, int N, int P>
struct value_type<math::Matrix<T,M,N>,P> 
	: value_type<math::VectorView<T,N>,P-1> {};

template <class T>
struct concept_arg<T, typename std::enable_if<math::is_matrix<T>::value>::type>
{
	typedef math::matrix_like<T> type;
};

} // namespace s3d

#include "matrix.hpp"

#endif

// $Id: matrix.h 3137 2010-09-21 03:37:15Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

