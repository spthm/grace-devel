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

#ifndef S3D_COLOR_YIQ_H
#define S3D_COLOR_YIQ_H

#include "fwd.h"
#include "coords.h"

namespace s3d { namespace color
{

template <class T> 
struct yiq_space/*{{{*/
{
	static const int dim = 3;
	typedef T value_type;
	typedef std::array<typename std::remove_const<T>::type,dim> 
		container_type;

	yiq_space() {}

	template <class U>
	yiq_space(const yiq_space<U> &that) 
		: m_coords(that.m_coords) {}

	yiq_space(T y, T i, T q)
		: m_coords((container_type){{y,i,q}}) {}

	template <class U>
	yiq_space &operator=(const yiq_space<U> &that) 
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
			T y, i, q;
		};
		container_type m_coords;
	};
};/*}}}*/

template <class T>
class YIQ
	: public coords<YIQ<T>,yiq_space<T>>
{
	typedef coords<YIQ<T>,yiq_space<T>> coords_base;
public:
	using coords_base::y;
	using coords_base::i;
	using coords_base::q;

	YIQ() {}
	YIQ(T y, T i, T q) 
		: coords_base(y, i, q) {}
	explicit YIQ(const radiance &r);

	template <class U>
	YIQ(const YIQ<U> &c) 
		: coords_base(c.y, c.i, c.q) {}

	operator radiance() const;
};

namespace traits
{
	template <class T>
	struct model<YIQ<T>>
	{
		static const color::model value = color::model::YIQ;
	};
}

}} // namespace s3d::color

namespace std
{
	template <class T>
	struct make_signed<s3d::color::YIQ<T>>
	{
		typedef s3d::color::YIQ<typename s3d::make_signed<T>::type> type;
	};
	template <class T>
	struct make_unsigned<s3d::color::YIQ<T>>
	{
		typedef s3d::color::YIQ<typename s3d::make_unsigned<T>::type> type;
	};
}

#include "yiq.hpp"

#endif

// $Id: yiq.h 2956 2010-08-13 02:33:38Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

