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

#ifndef S3D_IMAGE_FWD_H
#define S3D_IMAGE_FWD_H

#include "../color/fwd.h"
#include "../util/optional.h"

namespace s3d { 

namespace img
{

template <class T=void> class Image;

typedef Image<color::radiance> radiance;
typedef radiance image;

typedef Image<color::radiance_alpha> radiance_alpha;
typedef radiance_alpha timage;

class packed_image;

typedef Image<color::bgr8> bgr8;
typedef Image<color::bgr16> bgr16;

typedef Image<color::bgra8> bgra8;
typedef Image<color::bgra16> bgra16;

typedef Image<color::rgba8> rgba8;
typedef Image<color::rgba16> rgba16;

typedef Image<color::rgb8> rgb8;
typedef Image<color::rgb16> rgb16;

typedef Image<color::ya8> ya8;
typedef Image<color::ya16> ya16;

typedef Image<color::y8> y8;
typedef Image<color::y16> y16;

typedef Image<color::rgb> rgb;
typedef Image<color::rgba> rgba;
typedef Image<color::bgr> bgr;
typedef Image<color::bgra> bgra;
typedef Image<color::luminance> luminance;
typedef Image<color::ya> ya;
typedef Image<color::yiq> yiq;
typedef Image<color::cmy> cmy;

template <class C=void> class large_image;
typedef large_image<color::bgr8> large_bgr8;
typedef large_image<color::bgr16> large_bgr16;

typedef large_image<color::bgra8> large_bgra8;
typedef large_image<color::bgra16> large_bgra16;

typedef large_image<color::rgba8> large_rgba8;
typedef large_image<color::rgba16> large_rgba16;

typedef large_image<color::rgb8> large_rgb8;
typedef large_image<color::rgb16> large_rgb16;

typedef large_image<color::ya8> large_ya8;
typedef large_image<color::ya16> large_ya16;

typedef large_image<color::y8> large_y8;
typedef large_image<color::y16> large_y16;

using color::model;

class parameters;

template <class I=void> struct image_traits;
template <class I> struct is_image;

} // namespace img

using img::image_traits;


} // namespace s3d

#endif

// $Id: fwd.h 2956 2010-08-13 02:33:38Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

