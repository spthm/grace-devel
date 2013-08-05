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

#ifndef S3D_COLOR_FWD_H
#define S3D_COLOR_FWD_H

#include <tuple>
#include <cstdint>
#include "../math/r3/fwd.h"

namespace s3d { namespace color
{

template <class C=void>
struct color_traits;

using std::get;

enum channel_position
{
	FIRST,
	LAST
};

template <class C, channel_position POS=LAST>
struct alpha;

template <class T, int N> struct Radiance;
typedef Radiance<real,3> radiance;
typedef alpha<radiance> radiance_alpha;

template <class T> struct RGB;
typedef RGB<real> rgb;
typedef RGB<uint8_t> rgb8;
typedef RGB<uint16_t> rgb16;
typedef RGB<uint32_t> rgb32;

typedef alpha<RGB<real>> rgba;
typedef alpha<RGB<uint8_t>> rgba8;
typedef alpha<RGB<uint16_t>> rgba16;
typedef alpha<RGB<uint32_t>> rgba32;

typedef alpha<RGB<real>,FIRST> argb;
typedef alpha<RGB<uint8_t>,FIRST> argb8;
typedef alpha<RGB<uint16_t>,FIRST> argb16;
typedef alpha<RGB<uint32_t>,FIRST> argb32;

template <class T> struct BGR;
typedef BGR<real> bgr;
typedef BGR<uint8_t> bgr8;
typedef BGR<uint16_t> bgr16;
typedef BGR<uint32_t> bgr32;

typedef alpha<BGR<real>> bgra;
typedef alpha<BGR<uint8_t>> bgra8;
typedef alpha<BGR<uint16_t>> bgra16;
typedef alpha<BGR<uint32_t>> bgra32;

typedef alpha<BGR<real>,FIRST> abgr;
typedef alpha<BGR<uint8_t>,FIRST> abgr8;
typedef alpha<BGR<uint16_t>,FIRST> abgr16;
typedef alpha<BGR<uint32_t>,FIRST> abgr32;

template <class T> struct CMY;
typedef CMY<real> cmy;
typedef CMY<uint8_t> cmy8;
typedef CMY<uint16_t> cmy16;
typedef CMY<uint32_t> cmy32;

typedef alpha<CMY<real>> cmya;
typedef alpha<CMY<uint8_t>> cmya8;
typedef alpha<CMY<uint16_t>> cmya16;
typedef alpha<CMY<uint32_t>> cmya32;

typedef alpha<CMY<real>,FIRST> acmy;
typedef alpha<CMY<uint8_t>,FIRST> acmy8;
typedef alpha<CMY<uint16_t>,FIRST> acmy16;
typedef alpha<CMY<uint32_t>,FIRST> acmy32;

template <class T> struct YIQ;
typedef YIQ<real> yiq;
typedef YIQ<uint8_t> yiq8;
typedef YIQ<uint16_t> yiq16;
typedef YIQ<uint32_t> yiq32;

template <class T> struct HSV;
typedef HSV<real> hsv;
typedef HSV<uint8_t> hsv8;
typedef HSV<uint16_t> hsv16;
typedef HSV<uint32_t> hsv32;

typedef real luminance;
typedef uint8_t y8;
typedef uint16_t y16;
typedef uint32_t y32;

typedef alpha<real> ya;
typedef alpha<uint8_t> ya8;
typedef alpha<uint16_t> ya16;
typedef alpha<uint32_t> ya32;

typedef alpha<real,FIRST> ay;
typedef alpha<uint8_t,FIRST> ay8;
typedef alpha<uint16_t,FIRST> ay16;
typedef alpha<uint32_t,FIRST> ay32;

enum class model
{
	UNDEFINED,
	RADIANCE,
	GRAYSCALE,
	RGB,
	CMY,
	YIQ,
	HSV
};

}}

#endif

// $Id: fwd.h 2956 2010-08-13 02:33:38Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

