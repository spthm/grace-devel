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

#ifndef S3D_COLOR_CMY_H
#define S3D_COLOR_CMY_H

#include "fwd.h"
#include "coords.h"
#include "cast.h"

namespace s3d { namespace color
{

template <class T> 
struct cmy_space/*{{{*/
{
	static const int dim = 3;

	typedef T value_type;
	typedef std::array<typename std::remove_const<T>::type,dim> 
		container_type;

	cmy_space() {}

	template <class U>
	cmy_space(const cmy_space<U> &that) 
		: m_coords(that.m_coords) {}

	cmy_space(T c, T m, T y)
		: m_coords((container_type){{c,m,y}}) {}

	template <class U>
	cmy_space &operator=(const cmy_space<U> &that) 
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
			T c, m, y;
		};
		container_type m_coords;
	};
};/*}}}*/

template <class T>
class CMY
	: public coords<CMY<T>,cmy_space<T>>
{
	typedef coords<CMY<T>,cmy_space<T>> coords_base;
public:
	using coords_base::c;
	using coords_base::m;
	using coords_base::y;

	template <class U>
	struct rebind { typedef CMY<U> type; };

	CMY() {}
	CMY(T c, T m, T y) 
		: coords_base(c, m, y) {}

	template <class U>
	CMY(const CMY<U> &c) : coords_base(c.c, c.m, c.y) {}

	explicit CMY(const radiance &r);
	operator radiance() const;
};

namespace traits
{
	template <class T>
	struct model<CMY<T>>
	{
		static const color::model value = color::model::CMY;
	};
}

}} // namespace s3d::color

namespace std
{
	template <class T>
	struct make_signed<s3d::color::CMY<T>>
	{
		typedef s3d::color::CMY<typename s3d::make_signed<T>::type> type;
	};
	template <class T>
	struct make_unsigned<s3d::color::CMY<T>>
	{
		typedef s3d::color::CMY<typename s3d::make_unsigned<T>::type> type;
	};
}

#include "cmy.hpp"

#endif

// $Id: cmy.h 2956 2010-08-13 02:33:38Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

