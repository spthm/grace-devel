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

#ifndef S3D_IMAGE_IO_H
#define S3D_IMAGE_IO_H

#include "fwd.h"
#include "format/format.h"
#include "../color/rgb.h"

namespace s3d { namespace img {

template <class I, class...PARAMS> 
I load(const std::string &fname, PARAMS &&...params);

template <class I, class...PARAMS> 
I load(const uint8_t *data, size_t len, PARAMS &&...params);

template <class I, class...PARAMS>
I load(std::istream &in, PARAMS &&...params);

template <class...PARAMS> 
std::unique_ptr<Image<>> load(const std::string &fname, PARAMS &&...params);

template <class...PARAMS> 
std::unique_ptr<Image<>> load(const uint8_t *data, size_t len, PARAMS &&...params);

template <class...PARAMS>
std::unique_ptr<Image<>> load(std::istream &in, PARAMS &&...params);

template <image_format F, class I, class...PARAMS> 
void save(const std::string &fname, I &&img, PARAMS &&...params);

template <image_format F, class I, class...PARAMS> 
void save(std::ostream &out, I &&img, PARAMS &&...params);

template <class I, class...PARAMS> 
void save(const std::string &fname, I &&img, PARAMS &&...params);

template <image_format F> bool is_format(std::istream &in);
template <image_format F> bool is_format(const uint8_t *data, size_t len);
template <image_format F> bool is_format(const std::string &fname);

template <image_format F> bool handles_extension(const std::string &ext);

}} // s3d::img

#include "io.hpp"

#endif

// $Id: io.h 2930 2010-08-06 18:44:18Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

