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

#ifndef S3D_IMAGE_FORMAT_FORMAT_H
#define S3D_IMAGE_FORMAT_FORMAT_H

#include "../config.h"

namespace s3d { namespace img
{

enum image_format
{
    BITMAP=1,
    KRO,
#if HAS_JPEG
    JPEG,
#endif
#if HAS_PNG
    PNG,
#endif
};

const std::string &to_string(image_format fmt);

std::ostream &operator<<(std::ostream &out, const image_format &fmt);
std::istream &operator>>(std::istream &in, image_format &fmt);

template <image_format F> struct format_traits;

}} // namespace s3d::img

#endif

// $Id: format.h 2936 2010-08-08 03:31:03Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

