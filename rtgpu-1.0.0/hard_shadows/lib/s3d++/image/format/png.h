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

#ifndef S3D_IMAGE_PNG_H
#define S3D_IMAGE_PNG_H

#include "../config.h"

#if HAS_PNG

#include "format.h"
#include "../fwd.h"

namespace s3d { namespace img
{

class large_image_impl;

template <> struct format_traits<PNG>
{
	static bool is_format(std::istream &in);
	static bool handle_extension(const std::string &ext);

	static const bool has_transparency = true;
	static const bool is_lossless = true;

	static std::unique_ptr<Image<>> load(std::istream &in, 
									   const parameters &p);

	static void save(std::ostream &out, const Image<> &img,const parameters &p);
	static void save(std::ostream &out, Image<> &&img,const parameters &p);

	static auto create_large(const image_traits<> &traits, const r2::usize &s, 
							 const std::string &fname, const parameters &p)
		-> std::unique_ptr<large_image_impl>;

	static auto load_large(const std::string &fname, const parameters &p)
		-> std::unique_ptr<large_image_impl>;
};

}} // namespace s3d::img
#endif

#endif

// $Id: png.h 2977 2010-08-18 22:37:40Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

