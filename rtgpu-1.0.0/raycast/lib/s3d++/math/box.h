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

#ifndef S3D_MATH_BOX_H
#define S3D_MATH_BOX_H

#include "real.h"
#include <boost/operators.hpp>
#include "fwd.h"
#include "vector.h"
#include "../util/gcc.h"

namespace s3d { namespace math
{

template <class T, int D> struct box_coords/*{{{*/
{
	typedef T value_type;

	box_coords() {}
	box_coords(const Point<T,D> &o, const Size<T,D> &s) : origin(o), size(s) {}

	Point<T,D> origin;
	Size<T,D> size;
};/*}}}*/
template <class T> struct box_coords<T,1>/*{{{*/
{
	static const int dim = 1;
	typedef T value_type;

	// When C++0x's unrestricted unions gets implemented in gcc, we'll use
	// a union to implement the aliases, as it is with euclidean and
	// dimensional coordinates. Meanwhile...

	box_coords() 
		: x(origin.x)
		, w(origin.w) {}

	box_coords(const box_coords &that)
		: origin(that.origin)
		, size(that.size)
		, x(origin.x)
		, w(size.w) {}

	box_coords(const Point<T,dim> &o, const Size<T,dim> &s) 
		: origin(o)
		, size(s)
		, x(origin.x)
		, w(size.w) {}

	box_coords &operator=(const box_coords &that)
		{ origin = that.origin; size = that.size; return *this; }

	Point<T,dim> origin;
	Size<T,dim> size;

	T &x;
	T &w;

	void set_x(T _x) { x = _x; }
	T get_x() const { return x; }
	void set_w(T _w) { w = _w; }
	T get_w() const { return w; }
};/*}}}*/
template <class T> struct box_coords<T,2>/*{{{*/
{
	static const int dim = 2;
	typedef T value_type;
	box_coords() 
		: x(origin.x), y(origin.y)
		, w(size.w), h(size.h) {}
	box_coords(const box_coords &that)
		: origin(that.origin)
		, size(that.size)
		, x(origin.x), y(origin.y)
		, w(size.w), h(size.h) {}

	box_coords(const Point<T,dim> &o, const Size<T,dim> &s) 
		: origin(o)
		, size(s)
		, x(origin.x), y(origin.y)
		, w(size.w), h(size.h) {}

	box_coords &operator=(const box_coords &that)
		{ origin = that.origin; size = that.size; return *this; }

	Point<T,dim> origin;
	Size<T,dim> size;

	T &x,&y;
	T &w,&h;

	void set_x(T _x) { x = _x; }
	T get_x() const { return x; }
	void set_y(T _y) { y = _y; }
	T get_y() const { return y; }
	void set_w(T _w) { w = _w; }
	T get_w() const { return w; }
	void set_h(T _h) { h = _h; }
	T get_h() const { return h; }
};/*}}}*/
template <class T> struct box_coords<T,3>/*{{{*/
{
	static const int dim = 3;
	typedef T value_type;
	box_coords() 
		: x(origin.x), y(origin.y), z(origin.z)
		, w(size.w), h(size.h), d(size.d) {}
	box_coords(const box_coords &that)
		: origin(that.origin)
		, size(that.size)
		, x(origin.x), y(origin.y), z(origin.z)
		, w(size.w), h(size.h), d(size.d) {}
	box_coords(const Point<T,dim> &o, const Size<T,dim> &s) 
		: origin(o)
		, size(s)
		, x(origin.x), y(origin.y), z(origin.z)
		, w(size.w), h(size.h), d(size.d) {}

	box_coords &operator=(const box_coords &that)
		{ origin = that.origin; size = that.size; return *this; }

	Point<T,dim> origin;
	Size<T,dim> size;

	T &x,&y,&z;
	T &w,&h,&d;

	void set_x(T _x) { x = _x; }
	T get_x() const { return x; }
	void set_y(T _y) { y = _y; }
	T get_y() const { return y; }
	void set_z(T _z) { z = _z; }
	T get_z() const { return z; }
	void set_w(T _w) { w = _w; }
	T get_w() const { return w; }
	void set_h(T _h) { h = _h; }
	T get_h() const { return h; }
	void set_d(T _d) { d = _d; }
	T get_d() const { return d; }
};/*}}}*/
template <class T> struct box_coords<T,4>/*{{{*/
{
	static const int dim = 4;
	typedef T value_type;
	box_coords() 
		: x(origin.x), y(origin.y), z(origin.z), w(origin.w)
		, width(size.width), h(size.h), d(size.d), s(size.s) {}
	box_coords(const box_coords &that)
		: origin(that.origin)
		, size(that.size)
		, x(origin.x), y(origin.y), z(origin.z), w(origin.w)
		, width(size.width), h(size.h), d(size.d), s(size.s) {}
	box_coords(const Point<T,dim> &o, const Size<T,dim> &s) 
		: origin(o)
		, size(s)
		, x(origin.x), y(origin.y), z(origin.z), w(origin.w)
		, width(size.width), h(size.h), d(size.d), s(size.s) {}

	box_coords &operator=(const box_coords &that)
		{ origin = that.origin; size = that.size; return *this; }

	Point<T,dim> origin;
	Size<T,dim> size;

	T &x,&y,&z,&w;
	T &width,&h,&d,&s; // s = spassitude

	void set_x(T _x) { x = _x; }
	T get_x() const { return x; }
	void set_y(T _y) { y = _y; }
	T get_y() const { return y; }
	void set_z(T _z) { z = _z; }
	T get_z() const { return z; }
	void set_w(T _w) { w = _w; }
	T get_w() const { return w; }
	void set_width(T _width) { width = _width; }
	T get_width() const { return width; }
	void set_h(T _h) { h = _h; }
	T get_h() const { return h; }
	void set_d(T _d) { d = _d; }
	T get_d() const { return d; }
	void set_s(T _s) { s = _s; }
	T get_s() const { return s; }
};/*}}}*/

template <class T, int D>
struct Box 
	: boost::andable<Box<T,D>, 
	  boost::orable<Box<T,D>>>
	, box_coords<T,D>
	, operators
{
private:
#if GCC_VERSION < 40500
	struct dummy { int a; }; // for safe-bool idiom
#endif
	typedef box_coords<T,D> coords_base;
public:
	using coords_base::origin;
	using coords_base::size;

	Box() {}

	Box(const Point<T,D> &lower, const Point<T,D> &upper)
		: coords_base(lower, Size<T,D>(upper-lower)) {}

	Box(const Point<T,D> &o, const Size<T,D> &s)
		: coords_base(o, s) {}

	template <class U> Box(const Box<U,D> &r)
		: coords_base(r.origin, r.size) {}

	Box(const Box &r)
		: coords_base(r) {}

	template <class...ARGS, class
		= typename std::enable_if<sizeof...(ARGS)+1 == D*2>::type>
	Box(const T &x1, const ARGS &...args);

	Box &operator *=(const T &b);
	Box &operator /=(const T &b);
	Box &operator -=(const T &b);
	Box &operator +=(const T &b);

	Box &operator +=(const Vector<T,D> &v);
	Box &operator -=(const Vector<T,D> &v);

	Box &operator *=(const Size<T,D> &s);
	Box &operator /=(const Size<T,D> &s);
	Box &operator -=(const Size<T,D> &s);
	Box &operator +=(const Size<T,D> &s);

	Box &merge(const Point<T,D> &pt);

	Box &operator &=(const Box &that);
	Box &operator |=(const Box &that);

	Box &operator |=(const Point<T,D> &that);

	bool operator==(const Box &that) const;
	bool operator!=(const Box &that) const { return !operator==(that); }

#if GCC_VERSION < 40500
	operator int dummy::*() const { return is_zero()?NULL:&dummy::a; }
#else
	explicit operator bool() const { return !is_zero(); }
#endif

	bool operator!() const { return is_zero(); }

	Point<T,D> constrain(const Point<T,D> &pt) const;

	bool contains(const Point<T,D> &pt) const;
	bool contains(const Box<T,D> &bx) const;

	bool is_positive() const;
	bool is_negative() const;
	bool is_null() const;
	bool is_zero() const;

#if HAS_SERIALIZATION
private:
	friend class boost::serialization::access;

	template <class A>
	void serialize(A &ar, unsigned int version)
	{
		ar & origin;
		ar & size;
	}
#endif
};

template <class T, int D> 
Box<T,D> null_box();

template <class T, class U, int D> 
auto centered_box(const Point<T,D> &c, const Size<U,D> &s)
	-> Box<typename value_type<decltype(c-s/2)>::type, D>;

template <class T, int D> 
Box<T,D> ceil(const Box<T,D> &r);

template <class T, int D> 
Box<T,D> floor(const Box<T,D> &r);

template <class T, int D> 
Box<T,D> round(const Box<T,D> &r);

template <class U, class T, int D> 
Box<T,D> grow(const Box<T,D> &b, U d);

template <class T> 
Box<T,2> transpose(const Box<T,2> &b);

template <class T, int D> 
Point<T,D> centroid(const Box<T,D> &b);

template <class T, int D> 
T max_dim(const Box<T,D> &b);

template <class T, int D> 
T min_dim(const Box<T,D> &b);

template <class T> 
real aspect(const Box<T,2> &b);

template <class T> 
real aspect(const Box<T,3> &b);

template <class T, int D>
T surface_area(const Box<T,D> &b);

template <class T, int D>
int maximum_extent(const Box<T,D> &b);

template <class T, class U, int D>
bool overlap(const Box<T,D> &b1, const Box<U,D> &b2);

template <class T, int D> 
const Point<T,D> &lower(const Box<T,D> &b);

template <class T, int D, class... ARGS> 
Point<T,D> lower(const Box<T,D> &b1, const ARGS &...args);

template <class T, int D> 
Point<T,D> upper(const Box<T,D> &b);

template <class T, int D, class... ARGS> 
Point<T,D> upper(const Box<T,D> &b1, const ARGS &...args);

template <class T, int D> 
auto normalize(const Box<T,D> &_r)
	-> typename std::enable_if<std::is_integral<T>::value,Box<T,D>>::type;

template <class T, int D> 
auto normalize(const Box<T,D> &_r)
	-> typename std::enable_if<!std::is_integral<T>::value,Box<T,D>>::type;

template <class T, int D> 
std::ostream &operator<<(std::ostream &out, const Box<T,D> &r);

template <class T, int D> 
std::istream &operator>>(std::istream &in, Box<T,D> &r);

template <class T, int D> 
size_t hash_value(const Box<T,D> &r);

}} // namespace s3d::math

namespace std
{
	template <class T, int D>
	struct hash<s3d::math::Box<T,D>>
	{
		size_t operator()(s3d::math::Box<T,D> p) const
		{
			return hash_value(p);
		}
	};

	template <class T, int D>
	struct make_signed<s3d::math::Box<T,D>>
	{
		typedef s3d::math::Box<typename make_signed<T>::type,D> type;
	};
	template <class T,int D>
	struct make_unsigned<s3d::math::Box<T,D>>
	{
		typedef s3d::math::Box<typename make_unsigned<T>::type,D> type;
	};
}

#include "box.hpp"

#endif

// $Id: box.h 3173 2010-10-28 16:09:51Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

