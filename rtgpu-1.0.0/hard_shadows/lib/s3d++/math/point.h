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

#ifndef S3D_MATH_POINT_H
#define S3D_MATH_POINT_H

#include "../util/type_traits.h"
#include "real.h"
#include "fwd.h"
#include "coords.h"
#include "vector.h"
#include "size.h"

namespace s3d { namespace math
{

namespace detail
{
	struct is_point_tag {};
}

template<class T, int D> 
struct Point 
	: coords<Point<T,D>,euclidean_space<T,D>>
	, detail::is_point_tag
{
private:
	typedef coords<Point,euclidean_space<T,D>> coords_base;

public:
	Point() {}
	template <class DUMMY=int,
		class = typename std::enable_if<sizeof(DUMMY)!=0 && D!=RUNTIME>::type>
	explicit Point(const T &v);

	template <class U, int N=D> struct rebind { typedef Point<U,N> type; };

	Point(const dimension<1>::type &d, const T &v);

	explicit Point(const dimension<1>::type &d) : coords_base(d) {}

	Point(const Vector<T,D> &v);

	template <class... ARGS, class = 
		typename std::enable_if<D==RUNTIME || D==sizeof...(ARGS)+1>::type>
	Point(T c1, ARGS... cn) : coords_base(c1, cn...) {}

	template <class U> Point(const Point<U,D> &that);
	Point(const Point &that);

	using coords_base::begin;
	using coords_base::end;
};

template <class T>
struct is_point
{
	static const bool value = std::is_base_of<detail::is_point_tag,T>::value;
};

template <class T>
struct is_point<T&> : is_point<typename std::remove_cv<T>::type>
{
};

template <class T, int D> 
std::ostream &operator<<(std::ostream &out, const Point<T,D> &pt);
template <class T, int D> 
std::istream &operator>>(std::istream &in, Point<T,D> &pt);

template <class T, int D> 
const Point<T,D> &lower(const Point<T,D> &p);

template <class T, int D, class... ARGS> 
Point<T,D> lower(const Point<T,D> &p1, const ARGS &...args);

template <class T, int D> 
const Point<T,D> &upper(const Point<T,D> &p);

template <class T, int D, class... ARGS> 
Point<T,D> upper(const Point<T,D> &p1, const ARGS &...args);

template <class T, int D> 
Point<T,D> round(const Point<T,D> &p);

template <class T, int D> 
Point<T,D> rint(const Point<T,D> &p);

template <class T, int D> 
Point<T,D> ceil(const Point<T,D> &p);

template <class T, int D> 
Point<T,D> floor(const Point<T,D> &p);

template <class T, int D> 
Point<T,D> origin();

template <class T, int D>
size_t hash_value(const Point<T,D> &p);


template <class T>
struct point_like : concept<T> {};

} // namespace math

template <class T>
struct concept_arg<T, typename std::enable_if<math::is_point<T>::value>::type>
{
	typedef math::point_like<T> type;
};

} // namesapce s3d

namespace std
{
	template <class T, int D>
	struct hash<s3d::math::Point<T,D>>
	{
		size_t operator()(s3d::math::Point<T,D> p) const
		{
			return hash_value(p);
		}
	};

	template <class T, int D>
	struct make_signed<s3d::math::Point<T,D>>
	{
		typedef s3d::math::Point<typename s3d::make_signed<T>::type,D> type;
	};
	template <class T,int D>
	struct make_unsigned<s3d::math::Point<T,D>>
	{
		typedef s3d::math::Point<typename s3d::make_unsigned<T>::type,D> type;
	};
}

#include "point.hpp"

#endif

// $Id: point.h 3124 2010-09-09 19:03:33Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

