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

#ifndef IMAGE_KRO_H
#define IMAGE_KRO_H

#include "format.h"
#include "../../math/r2/size.h"
#include "../fwd.h"

namespace s3d { namespace img
{

class large_image_impl;

#pragma pack(push,1)
struct kro_header
{
	char sig[3]; // Magic identifier (KRO)
	uint8_t version;
	uint32_t width;
	uint32_t height;
	uint32_t depth; // 8 bits, 16 bits, 32 bits (float)
	uint32_t ncomp; // number of components, 3 (RGB) or 4 (RGBA)
};
#pragma pack(pop)


template <> struct format_traits<KRO>
{
	static bool is_format(std::istream &in);
	static bool handle_extension(const std::string &ext);
	static const bool has_transparency = true;
	static const bool is_lossless = true;

	static std::unique_ptr<Image<>> load(std::istream &in, const parameters &p);

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

// $Id: kro.h 2896 2010-08-01 01:30:44Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

