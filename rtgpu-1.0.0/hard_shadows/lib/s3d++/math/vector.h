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

#ifndef S3D_MATH_VECTOR_H
#define S3D_MATH_VECTOR_H

#include <array>
#include <vector>
#include "real.h"
#include "fwd.h"
#include "coords.h"
#include "size.h"
#include "r4/fwd.h"
#include "r2/fwd.h"
#include "../util/concepts.h"
#include "euclidean_space.h"

namespace s3d { namespace math
{


template <class T>
struct is_vector
{
	static const bool value = false;
};

template <class T>
struct is_vector<T&> : is_vector<typename std::remove_cv<T>::type>
{
};

template <class T, int N> 
struct Vector
	: coords<Vector<T,N>,euclidean_space<T,N>>
{
private:
	typedef coords<Vector,euclidean_space<T,N>> coords_base;

public:
	Vector() {} 

	template <class DUMMY=int,
		class = typename std::enable_if<sizeof(DUMMY)!=0 && N!=RUNTIME &&
				(N > 1)>::type>
	explicit Vector(const T &v);

	template <class U, int D=N> struct rebind { typedef Vector<U,D> type; };

	Vector(const dimension<1>::type &d, const T &v);

	explicit Vector(const dimension<1>::type &d) : coords_base(d) {}

	// defined in point.hpp
	Vector(const Point<T,N> &v);

	Vector(const Vector &p) = default;

	template <class U, int P>
	Vector(const Vector<U,P> &v) : coords_base(v) {}

	template <class U, int P>
	Vector(Vector<U,P> &&v) : coords_base(std::move(v)) {}

	template <class... ARGS, class = 
		typename std::enable_if<N==RUNTIME || N==sizeof...(ARGS)+1>::type>
	Vector(T v1, ARGS ...vn) 
		: coords_base(v1, vn...) {}

	template <template<class,int> class V, class U, int P,
		class = typename std::enable_if<is_vector<V<U,P>>::value>::type>
	Vector &operator =(const V<U,P> &that);

	template <template<class,int> class V, class U, int P,
		class = typename std::enable_if<is_vector<V<U,P>>::value>::type>
	Vector &operator =(V<U,P> &&that);

	using coords_base::size;
	using coords_base::begin;
	using coords_base::end;

	std::size_t cols() const { return size(); }
};

template <class T, int N>
struct is_vector<Vector<T,N>>
{
	static const bool value = true;
};


template <class T>
struct vector_like : concept<T> {};

} // namespace math

template <class T>
struct concept_arg<T, typename std::enable_if<math::is_vector<T>::value>::type>
{
	typedef math::vector_like<T> type;
};

} //namespace s3d

namespace std
{
	template <class T, int D>
	struct make_signed<s3d::math::Vector<T,D>>
	{
		typedef s3d::math::Vector<typename s3d::make_signed<T>::type,D> type;
	};
	template <class T, int D>
	struct make_unsigned<s3d::math::Vector<T,D>>
	{
		typedef s3d::math::Vector<typename s3d::make_unsigned<T>::type,D> type;
	};
}

#include "vector.hpp"

#endif

	// $Id: vector.h 3076 2010-08-31 05:40:23Z rodolfo $
	// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
	// vi: ai sw=4 ts=4

