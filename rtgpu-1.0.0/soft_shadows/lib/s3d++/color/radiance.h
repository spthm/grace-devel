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

#ifndef S3D_COLOR_RADIANCE_H
#define S3D_COLOR_RADIANCE_H

#include "../math/vector.h"
#include "coords.h"
#include "fwd.h"
#include "traits.h"
#include "compositing.h"

namespace s3d { namespace color
{

using math::map;
using math::unmap;

template <class T, int D> 
struct Radiance
	: coords<Radiance<T,D>,math::euclidean_space<T,D>>
	, compositing_operators<Radiance<T,D>>
{
private:
	typedef coords<Radiance<T,D>,math::euclidean_space<T,D>> coords_base;
public:
	using coords_base::begin;
	using coords_base::end;

	Radiance() {} // so that it can be in a std::tuple
	explicit Radiance(const math::dimension<1>::type &d) : coords_base(d) {}

	Radiance(const Vector<T,D> &v) : coords_base(v) {}

	template <class U, class =
		typename std::enable_if<std::is_same<Radiance,radiance>::value &&
					std::is_arithmetic<U>::value>::type>
	Radiance(const U &c);

	template <class U>
	Radiance(const Radiance<U,D> &c)
	{
		std::copy(c.begin(), c.end(), begin());
	}

	Radiance(const Radiance &that) = default;

	template <class... ARGS, class =
		typename std::enable_if<sizeof...(ARGS)+1 == D>::type>
	Radiance(T c1, ARGS... cn);
};

namespace traits
{
	template <class T, int D>
	struct model<Radiance<T,D>>
	{
		static const color::model value = color::model::RADIANCE;
	};
}

} // namespace color

namespace math
{
	template <class T, int D>
	struct is_vector<color::Radiance<T,D>>
	{
		static const bool value = true;
	};

} // namespace math

} // s3d

namespace std
{
	template <class T, int D>
	struct make_signed<s3d::color::Radiance<T,D>>
	{
		typedef s3d::color::Radiance<typename s3d::make_signed<T>::type,D> type;
	};
	template <class T, int D>
	struct make_unsigned<s3d::color::Radiance<T,D>>
	{
		typedef s3d::color::Radiance<typename s3d::make_unsigned<T>::type,D> type;
	};
}

#include "radiance.hpp"

#endif

// $Id: radiance.h 2973 2010-08-18 13:58:40Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

