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

#ifndef S3D_COLOR_RGB_H
#define S3D_COLOR_RGB_H

#include "fwd.h"
#include "radiance.h"
#include "compositing.h"
#include "coords.h"
#include "cast.h"
#include "../math/v4f.h"

namespace s3d { namespace color
{

using math::v4f;

template <class T> 
struct rgb_space/*{{{*/
{
	static const int dim = 3;
	static const bool invert = false;

	typedef T value_type;
	typedef std::array<typename std::remove_const<T>::type,dim> 
		container_type;

	typedef rgb_space coord_def_type;

	rgb_space() {}

	template <class U>
	rgb_space(const rgb_space<U> &that) 
		: m_coords(that.m_coords) {}

	rgb_space(T _r, T _g, T _b)
		: m_coords((container_type){{_r,_g,_b}}) {}

	template <class U>
	rgb_space &operator=(const rgb_space<U> &that) 
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
			T r, g, b;
		};
		container_type m_coords;
	};
};/*}}}*/

template <class T>
class RGB
	: public coords<RGB<T>,rgb_space<T>>
	, public compositing_operators<RGB<T>>
{
	typedef coords<RGB<T>,rgb_space<T>> coords_base;
public:
	using coords_base::r;
	using coords_base::g;
	using coords_base::b;

	RGB() {}
	RGB(T r, T g, T b) 
		: coords_base(r, g, b) {}

	template <class U>
	RGB(const RGB<U> &c) : coords_base(c.r, c.g, c.b) {}

	explicit RGB(const radiance &r) 
		: coords_base(map<T>(r.x), map<T>(r.y), map<T>(r.z)) {}

	operator radiance() const 
		{ return radiance(unmap(r),unmap(g),unmap(b)); }
};

template <class T> 
struct bgr_space/*{{{*/
{
	static const int dim = 3;
	static const bool invert = true;

	typedef T value_type;
	typedef std::array<typename std::remove_const<T>::type,dim> 
		container_type;

	bgr_space() {}

	template <class U>
	bgr_space(const bgr_space<U> &that) 
		: m_coords(that.m_coords) {}

	bgr_space(T _b, T _g, T _r)
		: m_coords((container_type){{_b,_g,_r}}) {}

	template <class U>
	bgr_space &operator=(const bgr_space<U> &that) 
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
			T b, g, r;
		};
		container_type m_coords;
	};
};/*}}}*/

template <class T>
class BGR
	: public coords<BGR<T>,bgr_space<T>>
	, public compositing_operators<BGR<T>>
{
	typedef coords<BGR<T>,bgr_space<T>> coords_base;
public:
	using coords_base::b;
	using coords_base::g;
	using coords_base::r;

	BGR() {}
	BGR(T b, T g, T r) 
		: coords_base(b, g, r) {}
	explicit BGR(const radiance &r) 
		: coords_base(map<T>(r.z), map<T>(r.y), map<T>(r.x)) {}

	template <class U>
	BGR(const BGR<U> &c) : coords_base(c.b, c.g, c.r) {}

	operator radiance() const 
		{ return radiance(unmap(r),unmap(g),unmap(b)); }
};

template <class T>
struct cast_def<RGB<T>, BGR<T>>
{
	static BGR<T> map(const RGB<T> &c)
	{
		return {c.b, c.g, c.r};
	}
};

template <class FROM, class TO>
struct cast_def<RGB<FROM>, BGR<TO>>
{
	static BGR<TO> map(const RGB<FROM> &c)
	{
		return color_cast<BGR<TO>>(color_cast<RGB<TO>>(c));
	}
};

template <class T>
struct cast_def<BGR<T>, RGB<T>>
{
	static RGB<T> map(const BGR<T> &c)
	{
		return {c.r, c.g, c.b};
	}
};

template <class FROM, class TO>
struct cast_def<BGR<FROM>, RGB<TO>>
{
	static RGB<TO> map(const BGR<FROM> &c)
	{
		return color_cast<RGB<TO>>(color_cast<BGR<TO>>(c));
	}
};

template <class FROM>
struct cast_def<FROM, v4f>
{
	static v4f map(const FROM &c)
	{
		auto rgba = color_cast<alpha<RGB<typename traits::value_type<FROM>::type>>>(c);
		return color_cast<v4f>(rgba);
	}
};

template <class TO>
struct cast_def<v4f, TO>
{
	static TO map(const v4f &c)
	{
		auto rgba = color_cast<alpha<RGB<typename traits::value_type<TO>::type>>>(c);
		return color_cast<TO>(rgba);
	}
};

template <class FROM, channel_position P>
struct cast_def<alpha<RGB<FROM>,P>, v4f>
{
	static v4f map(const alpha<RGB<FROM>,P> &c)
	{
		return v4f(unmap(c.r), unmap(c.g), unmap(c.b), unmap(c.a));
	}
};

template <class TO, channel_position P>
struct cast_def<v4f, alpha<RGB<TO>,P>>
{
	static alpha<RGB<TO>,P> map(const v4f &c)
	{
		float a[4];
		c.convert_to(a);

		return alpha<RGB<TO>,P>({math::map<TO>(a[0]), math::map<TO>(a[1]),
								 math::map<TO>(a[2])}, math::map<TO>(a[3]));
	}
};

// disambiguation
template <class TO, channel_position P>
struct cast_def<v4f, alpha<TO,P>>
{
	static alpha<TO,P> map(const v4f &c)
	{
		auto rgba = color_cast<alpha<RGB<typename traits::value_type<TO>::type>,P>>(c);
		return color_cast<alpha<TO,P>>(rgba);
	}
};
template <class FROM, channel_position P>
struct cast_def<alpha<FROM,P>, v4f>
{
	static v4f map(const alpha<FROM,P> &c)
	{
		auto rgba = color_cast<alpha<RGB<typename traits::value_type<alpha<FROM,P>>::type>>>(c);
		return color_cast<v4f>(rgba);
	}
};


namespace traits
{
	template <> 
	struct has_alpha<v4f> : std::true_type {};

	template <> 
	struct model<v4f>
	{
		static const color::model value = color::model::RGB;
	};

	template <> 
	struct dim<v4f>
	{
		static const size_t value = 4;
	};

	template <> 
	struct bpp<v4f>
	{
		static const size_t value = 4*sizeof(float);
	};

	template <>
	struct value_type<v4f>
	{
		typedef float type;
	};

	template <>
	struct is_floating_point<v4f>
	{
		static const bool value = true;
	};

	template <>
	struct is_integral<v4f>
	{
		static const bool value = false;
	};

	template <class T>
	struct model<RGB<T>>
	{
		static const color::model value = color::model::RGB;
	};

	template <class T>
	struct model<BGR<T>>
	{
		static const color::model value = color::model::RGB;
	};
}

}} // namespace s3d::color

namespace std
{
	template <class T>
	struct make_signed<s3d::color::RGB<T>>
	{
		typedef s3d::color::RGB<typename s3d::make_signed<T>::type> type;
	};
	template <class T>
	struct make_unsigned<s3d::color::RGB<T>>
	{
		typedef s3d::color::RGB<typename s3d::make_unsigned<T>::type> type;
	};

	template <class T>
	struct make_signed<s3d::color::BGR<T>>
	{
		typedef s3d::color::BGR<typename s3d::make_signed<T>::type> type;
	};
	template <class T>
	struct make_unsigned<s3d::color::BGR<T>>
	{
		typedef s3d::color::BGR<typename s3d::make_unsigned<T>::type> type;
	};
}

#endif

// $Id: rgb.h 3077 2010-08-31 05:40:40Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

