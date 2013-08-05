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

#ifndef S3D_COLOR_CAST_H
#define S3D_COLOR_CAST_H

#include "traits.h"
#include "../util/type_traits.h"
#include "../util/concepts.h"
#include "../math/real.h"
#include "radiance.h"

namespace s3d { namespace color
{

template <class FROM, class TO> struct cast_def;

namespace detail
{
	// precisamos criar um nível de indireção p/ só chamar o cast_def
	// se os tipos forem colorspaces, p/ evitar warnings caso não sejam
	template <class FROM, class TO, bool B=is_colorspace<FROM>::value &&
		                                   is_colorspace<TO>::value>
	struct color_cast_ret
	{
		typedef decltype(cast_def<FROM,TO>::map(std::declval<const FROM>())) type;
	};

	template <class FROM, class TO>
	struct color_cast_ret<FROM,TO,false>
	{
	};
}

template <class FROM, class TO, class EN=void>
struct can_cast
{
	static const bool value = false;
};

template <class FROM, class TO>
struct can_cast<FROM,TO,
	typename std::enable_if<sizeof(typename detail::color_cast_ret<FROM,TO>::type)>::type>
{
	static const bool value = true;
};

template <class TO, class FROM>
auto color_cast(const FROM &c)
	-> typename detail::color_cast_ret<FROM,TO>::type
{
	return cast_def<FROM,TO>::map(c);
}

}

using color::color_cast;

} // namespace s3d::color

using s3d::color::color_cast;

#include "cast_def.h"

#endif

// $Id: cast.h 2955 2010-08-12 23:48:06Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

