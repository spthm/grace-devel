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

#ifndef S3D_MATH_SIZE_H
#define S3D_MATH_SIZE_H

#include <array>
#include <functional>
#include "../util/type_traits.h" 
#include "real.h"
#include "fwd.h"
#include "coords.h"
#include "size_space.h"

namespace s3d { namespace math
{

template <class T>
struct is_size
{
	static const bool value = false;
};

template <class T>
struct is_size<T&> : is_size<typename std::remove_cv<T>::type>
{
};

template <class T, int D> struct Vector;

template<class T, int D> 
struct Size 
	: coords<Size<T,D>,size_space<T,D>>
{
private:
	typedef coords<Size,size_space<T,D>> coords_base;
public:
	template <class U, int N=D> struct rebind {typedef Size<U,N> type;};

	Size() {}

	template <class... ARGS> 
	Size(T c1, ARGS... cn) : coords_base(c1, cn...) {}

	explicit Size(const T &v);
	template <class U> Size(const Size<U,D> &that);
	Size(const Size &that);
	// Defined in vector.hpp
	template <class U> explicit Size(const Vector<U,D> &v);

	Size(const dimension<1>::type &d, const T &v);

	explicit Size(const dimension<1>::type &d) : coords_base(d) {}

	using coords_base::size;
	using coords_base::begin;
	using coords_base::end;

	Size operator -();

	Size &operator +=(T d);
	Size &operator -=(T d);
	Size &operator /=(T d);
	Size &operator *=(T d);

	template <class U, int M, class = 
		typename std::enable_if<M==D || M==RUNTIME || D==RUNTIME>::type>
	Size &operator+=(const Size<U,M> &s);

	template <class U, int M, class =
		typename std::enable_if<M==D || M==RUNTIME || D==RUNTIME>::type>
	Size &operator-=(const Size<U,M> &s);

	template <class U, int M, class =
		typename std::enable_if<M==D || M==RUNTIME || D==RUNTIME>::type>
	Size &operator*=(const Size<U,M> &s);

	template <class U, int M, class =
		typename std::enable_if<M==D || M==RUNTIME || D==RUNTIME>::type>
	Size &operator/=(const Size<U,M> &s);

	template <class U, int M, class = 
		typename std::enable_if<M==D || M==RUNTIME || D==RUNTIME>::type>
	Size &operator+=(const Vector<U,M> &s);

	template <class U, int M, class =
		typename std::enable_if<M==D || M==RUNTIME || D==RUNTIME>::type>
	Size &operator-=(const Vector<U,M> &s);

	bool is_positive() const;
	bool is_negative() const;
	bool is_zero() const;
};

template <class T, int N>
struct is_size<Size<T,N>>
{
	static const bool value = true;
};

template <class T>
struct size_like : concept<T> {};

template <class T, int D> 
const Size<T,D> &lower(const Size<T,D> &p);

template <class T, int D, class... ARGS> 
Size<T,D> lower(const Size<T,D> &p1, const ARGS &...args);

template <class T, int D> 
const Size<T,D> &upper(const Size<T,D> &p);

template <class T, int D, class... ARGS> 
Size<T,D> upper(const Size<T,D> &p1, const ARGS &...args);

template <class T, int D> 
Size<T,D> round(const Size<T,D> p);

template <class T, int D> 
std::ostream &operator<<(std::ostream &out, const Size<T,D> &sz);
template <class T, int D> 
std::istream &operator>>(std::istream &in, Size<T,D> &sz);

template <class T, int D>
size_t hash_value(const Size<T,D> &p);

} // namespace math

template <class T>
struct concept_arg<T, typename std::enable_if<math::is_size<T>::value>::type>
{
	typedef math::size_like<T> type;
};

} // namespace s3d::math

namespace std
{
	template <class T, int D>
	struct hash<s3d::math::Size<T,D>>
	{
		size_t operator()(s3d::math::Size<T,D> p) const
		{
			return hash_value(p);
		}
	};

	template <class T, int D>
	struct make_signed<s3d::math::Size<T,D>>
	{
		typedef s3d::math::Size<typename make_signed<T>::type,D> type;
	};
	template <class T, int D>
	struct make_unsigned<s3d::math::Size<T,D>>
	{
		typedef s3d::math::Size<typename make_unsigned<T>::type,D> type;
	};
}


#include "size.hpp"

#endif

// $Id: size.h 3132 2010-09-15 00:18:39Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

