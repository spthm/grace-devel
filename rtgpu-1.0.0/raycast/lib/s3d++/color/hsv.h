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

#ifndef S3D_COLOR_HSV_H
#define S3D_COLOR_HSV_H

#include "../math/r3/point.h"
#include "radiance.h"
#include "coords.h"

namespace s3d { namespace color
{

template <class T> 
struct hsv_space/*{{{*/
{
	static const int dim = 3;
	typedef T value_type;
	typedef std::array<typename std::remove_const<T>::type,dim> 
		container_type;

	hsv_space() {}

	template <class U>
	hsv_space(const hsv_space<U> &that) 
		: m_coords(that.m_coords) {}

	hsv_space(T h, T s, T v)
		: m_coords((container_type){{h,s,v}}) {}

	template <class U>
	hsv_space &operator=(const hsv_space<U> &that) 
	{
		m_coords = that.m_coords;
		return *this;
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

	union
	{
		struct
		{
			T h, s, v;
		};
		container_type m_coords;
	};
};/*}}}*/

template <class T>
class HSV
	: public coords<HSV<T>,hsv_space<T>>
{
	typedef coords<HSV<T>,hsv_space<T>> coords_base;
public:
	using coords_base::h;
	using coords_base::s;
	using coords_base::v;

	HSV() {}
	HSV(T h, T s, T v) 
		: coords_base(h, s, v) {}
	explicit HSV(const radiance &r);

	template <class U>
	HSV(const HSV<U> &c) : coords_base(c.h, c.s, c.v) {}

	operator radiance() const;
};

namespace traits
{
	template <class T>
	struct model<HSV<T>>
	{
		static const color::model value = color::model::HSV;
	};
}

}} // namespace s3d::color

namespace std
{
	template <class T>
	struct make_signed<s3d::color::HSV<T>>
	{
		typedef s3d::color::HSV<typename s3d::make_signed<T>::type> type;
	};
	template <class T>
	struct make_unsigned<s3d::color::HSV<T>>
	{
		typedef s3d::color::HSV<typename s3d::make_unsigned<T>::type> type;
	};
}

#include "hsv.hpp"

#endif

// $Id: hsv.h 2209 2009-06-01 02:54:31Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

