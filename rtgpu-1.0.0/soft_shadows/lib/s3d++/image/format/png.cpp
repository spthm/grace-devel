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

// $Id: png.cpp 3098 2010-09-02 02:41:00Z rodolfo $
// vim: nocp:ci:sts=4:fdm=marker:fmr={{{,}}}
// vi: ai:sw=4:ts=8

#include "../pch.h"
#include "../image.h"
#include "../cast.h"
#include "../large_image.h"
#include "../../color/luminance.h"
#include "../../color/rgb.h"
#include "../params.h"
#include "util.h"
#include "png.h"

#if HAS_PNG

namespace p = std::placeholders;

namespace s3d { namespace img
{

namespace 
{
const image_traits<> &get_traits(int dim, int bit_depth)/*{{{*/
{
	switch(dim)
	{
	case 1:
		switch(bit_depth)
		{
		case 8:
			return y8::traits();
		case 16:
			return y16::traits();
		default:
			throw std::runtime_error("PNG: bit depth not supported");
		}
		break;
	case 3:
		switch(bit_depth)
		{
		case 8:
			return rgb8::traits();
		case 16:
			return rgb16::traits();
		default:
			throw std::runtime_error("PNG: bit depth not supported");
		}
		break;
	case 4:
		switch(bit_depth)
		{
		case 8:
			return rgba8::traits();
		case 16:
			return rgba16::traits();
		default:
			throw std::runtime_error("PNG: bit depth not supported");
		}
		break;
	case 2:
		switch(bit_depth)
		{
		case 8:
			return ya8::traits();
		case 16:
			return ya16::traits();
		default:
			throw std::runtime_error("PNG: bit depth not supported");
		}
		break;
	default:
		throw std::runtime_error("PNG: Image type not supported");
	}
}/*}}}*/

void stream_read(png_structp pp, png_bytep data, png_size_t length)/*{{{*/
{
	std::istream &in = *reinterpret_cast<std::istream *>(png_get_io_ptr(pp));
	in.read((char *)data, length);
	if(!in)
		png_error(pp, "Cannot read from file");
}/*}}}*/
void stream_write(png_structp pp, png_bytep data, png_size_t length)/*{{{*/
{
	std::ostream &out = *reinterpret_cast<std::ostream *>(png_get_io_ptr(pp));
	out.write((char *)data, length);
	if(!out)
		png_error(pp, "Cannot write to file");
}/*}}}*/
void stream_flush(png_structp pp)/*{{{*/
{
	std::ostream &out = *reinterpret_cast<std::ostream *>(png_get_io_ptr(pp));
	out.flush();
}/*}}}*/

void save_png(std::ostream &out, const Image<> &img, const parameters &p)/*{{{*/
{
	png_structp pp = NULL;
	png_infop info = NULL;

	try
	{
		pp = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL,NULL,NULL);

		if(!pp)
			throw std::runtime_error("Error creating png write struct");

		info = png_create_info_struct(pp);
		if(!info)
			throw std::runtime_error("Error creating png info struct");

		png_set_write_fn(pp, &out, &stream_write, &stream_flush);

		if(setjmp(png_jmpbuf(pp)))
			throw std::runtime_error("Error writing png");

		int color_type;
		switch(img.traits().dim)
		{
		case 1:
			color_type = PNG_COLOR_TYPE_GRAY;
			break;
		case 2:
			color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
			break;
		case 3:
			color_type = PNG_COLOR_TYPE_RGB;
			break;
		case 4:
			color_type = PNG_COLOR_TYPE_RGBA;
			break;
		default:
			assert(false);
			color_type = -1;
			break;
		}

		png_set_IHDR(pp, info, img.w(), img.h(), img.traits().depth, color_type,
					 PNG_INTERLACE_NONE, 
					 PNG_COMPRESSION_TYPE_DEFAULT,
					 PNG_FILTER_TYPE_DEFAULT);

		int xform;
		if(img.traits().depth > 8)
			xform = PNG_TRANSFORM_SWAP_ENDIAN;
		else
			xform = PNG_TRANSFORM_IDENTITY;

		int row_stride = img.row_stride();
		std::vector<png_bytep> rows;
		rows.reserve(img.h());
		for(size_t i=0; i<img.h(); ++i)
			rows.push_back((png_bytep)(img.data() + i*row_stride));

		png_set_rows(pp, info, &rows[0]);
		png_write_png(pp, info, xform, NULL);

		png_destroy_write_struct(&pp, &info);
	}
	catch(...)
	{
		png_destroy_write_struct(&pp, &info);
		throw;
	}
}/*}}}*/

template <class I>
void save_img(std::ostream &out, I &&img, const parameters &p)/*{{{*/
{
	if(img.traits().depth == 8)
	{
		if(img.traits().has_alpha)
		{
			if(img.traits().model == model::GRAYSCALE)
				save_png(out, *image_cast_ptr<ya8>(std::forward<I>(img)), p);
			else
				save_png(out, *image_cast_ptr<rgba8>(std::forward<I>(img)), p);
		}
		else
		{
			if(img.traits().model == model::GRAYSCALE)
				save_png(out, *image_cast_ptr<y8>(std::forward<I>(img)), p);
			else
				save_png(out, *image_cast_ptr<rgb8>(std::forward<I>(img)), p);
		}
	}
	else
	{
		if(img.traits().has_alpha)
		{
			if(img.traits().model == model::GRAYSCALE)
				save_png(out, *image_cast_ptr<ya16>(std::forward<I>(img)), p);
			else
				save_png(out, *image_cast_ptr<rgba16>(std::forward<I>(img)), p);
		}
		else
		{
			if(img.traits().model == model::GRAYSCALE)
				save_png(out, *image_cast_ptr<y16>(std::forward<I>(img)), p);
			else
				save_png(out, *image_cast_ptr<rgb16>(std::forward<I>(img)), p);
		}
	}
}/*}}}*/
}

bool format_traits<PNG>::is_format(std::istream &in)/*{{{*/
{
	if(!in)
		throw std::runtime_error("Bad stream state");

	int pos = in.tellg();

	char sig[8];
	in.read(sig, sizeof(sig));
	if(!in || in.gcount() != sizeof(sig))
	{
		in.seekg(pos, std::ios::beg);
		in.clear();
		return false;
	}

	in.seekg(pos, std::ios::beg);

	return png_check_sig((png_bytep)sig, sizeof(sig)) ? true : false;
}/*}}}*/
bool format_traits<PNG>::handle_extension(const std::string &ext)/*{{{*/
{
	return boost::to_upper_copy(ext) == "PNG";
}/*}}}*/

std::unique_ptr<Image<>> format_traits<PNG>::load(std::istream &in,/*{{{*/
								const parameters &p)
{
	char sig[8];
	in.read(sig, sizeof(sig));

	if(!png_check_sig((png_bytep)sig, sizeof(sig)))
		throw std::runtime_error("Image is not a PNG");

	png_structp pp = NULL;
	png_infop info = NULL;

	try
	{
		pp = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL,NULL,NULL);

		if(!pp)
			throw std::runtime_error("Image is not a PNG");

		png_set_sig_bytes(pp,sizeof(sig));

		info = png_create_info_struct(pp);
		if(!info)
			throw std::runtime_error("PNG: error creating info structure");

		if(setjmp(png_jmpbuf(pp)))
			throw std::runtime_error("PNG: error reading image");

		png_set_read_fn(pp, &in, &stream_read);

		png_read_info(pp, info);

		int bit_depth = png_get_bit_depth(pp, info),
			color_type = png_get_color_type(pp,info),
			w = png_get_image_width(pp,info),
			h = png_get_image_height(pp,info);
#if 0
		png_get_IHDR(pp, info, (png_uint_32 *)&w, 
					 (png_uint_32 *)&h,
					 &bit_depth, &color_type, NULL, NULL, NULL);
#endif

		// Transforma color indexada em RGB
		if(color_type & PNG_COLOR_MASK_PALETTE)
		{
			png_set_palette_to_rgb(pp);
			color_type = PNG_COLOR_TYPE_RGB;
		}

		// Tem informação de alfa? Converte p/ RGBA
		if(png_get_valid(pp, info, PNG_INFO_tRNS))
		{
			png_set_tRNS_to_alpha(pp);
			color_type |= PNG_COLOR_MASK_ALPHA;
		}

		if((color_type & PNG_COLOR_MASK_ALPHA) && p[background_color])
		{
			auto bg = any_cast<color::radiance>(p[background_color]);
			auto rgb = color_cast<color::rgb16>(bg);

			png_color_16 bg16;
			bg16.red = rgb.r;
			bg16.green = rgb.g;
			bg16.blue = rgb.b;

			png_set_background(pp, &bg16, PNG_BACKGROUND_GAMMA_SCREEN, 0, 1.0);
			color_type &= ~PNG_COLOR_MASK_ALPHA;
		}

		if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		{
			png_set_expand_gray_1_2_4_to_8(pp);
			bit_depth = 8;
		}

		if(bit_depth > 8)
			png_set_swap(pp);

		int pixel_dim = 0;
		if(color_type == PNG_COLOR_TYPE_GRAY || 
		   color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
		{
			pixel_dim = 1;
		}
		else if(color_type == PNG_COLOR_TYPE_RGB || 
				color_type == PNG_COLOR_TYPE_RGB_ALPHA)
		{
			pixel_dim = 3;
		}
		else
		{
			assert(false);
			pixel_dim = 3;
		}

		if(color_type & PNG_COLOR_MASK_ALPHA)
			++pixel_dim;

		std::unique_ptr<Image<>> img = get_traits(pixel_dim, bit_depth)
											.create_image({w,h});

		int row_stride = img->row_stride();

		std::vector<png_bytep> rows;
		rows.reserve(h);
		for(int i=0; i<h; ++i)
			rows.push_back((png_bytep)(img->data() + i*row_stride));

		png_read_image(pp, &rows[0]);
		png_read_end(pp, NULL);

		png_destroy_read_struct(&pp, &info, NULL);

		return std::move(img);
	}
	catch(...)
	{
		png_destroy_read_struct(&pp, &info, NULL);
		throw;
	}
}/*}}}*/

void format_traits<PNG>::save(std::ostream &out, const Image<> &img,/*{{{*/
							  const parameters &p)
{
	save_img(out, img, p);
}/*}}}*/
void format_traits<PNG>::save(std::ostream &out, Image<> &&img,/*{{{*/
							  const parameters &p)
{
	save_img(out, std::move(img), p);
}/*}}}*/

auto format_traits<PNG>::load_large(const std::string &fname, /*{{{*/
									const parameters &p)
	-> std::unique_ptr<large_image_impl>
{
	std::ifstream in(fname);
	if(!in)
		throw std::runtime_error("Error opening "+fname);

	auto img = load(in, p);

	return make_unique<default_large_image_impl>(std::move(img), fname, p,
												 false);
}/*}}}*/

auto format_traits<PNG>::create_large(const image_traits<> &traits, /*{{{*/
									  const r2::usize &s, 
									  const std::string &fname,
									  const parameters &p)
	-> std::unique_ptr<large_image_impl>
{
	return make_unique<default_large_image_impl>(traits.create_image(s), 
												 fname, p, true);
}/*}}}*/

}} // namespace s3d::img

#endif


// $Id: png.cpp 3098 2010-09-02 02:41:00Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

