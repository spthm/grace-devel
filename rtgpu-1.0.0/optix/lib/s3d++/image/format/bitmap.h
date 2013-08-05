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

#ifndef IMAGE_BITMAP_H
#define IMAGE_BITMAP_H

#include "format.h"
#include "../../math/r2/size.h"
#include "../fwd.h"

namespace s3d { namespace img
{

class large_image_impl;

#pragma pack(push,1)
struct bmp_header/*{{{*/
{
    unsigned char type[2];	// Magic identifier (BM)
    uint32_t size;			// File size in bytes
    uint16_t reserved1, 
			 reserved2;
    uint32_t offset;		// Offset to image data, bytes
};/*}}}*/
struct bmp_dib_v3_header/*{{{*/
{
    uint32_t size;          // Header size in bytes (40)
    int32_t width,height;  // Width and height of image 
    uint16_t planes;        // Number of colour planes, must be 1
    uint16_t bits;          // Bits per pixel (1,4,8,16,24,32)
    uint32_t compression;   // Compression type:
							//	0 - BI_RGB - no compression
							//	1 - BI_RLE8 - 8 bit RLE
							//	2 - BI_RLE4 - 4 bit RLE
							//	3 - BI_BITFIELDS - bit field (?)
						    //  4 - BI_JPEG - bitmap contains a jpeg image
						    //  5 - BI_PNG - bitmap contains a png image
    uint32_t imagesize;		// Image size in bytes (raw), can be zero for
                            // compression == BI_RGB
    int32_t xresolution,    // pixels per meter
    		yresolution;    // ditto         
    uint32_t ncolours;			// Number of colours (0 == 2^bits)
    uint32_t importantcolours;	// Important colours (0 == all colors)
};/*}}}*/
#pragma pack(pop)

template <> struct format_traits<BITMAP>
{
	static bool is_format(std::istream &in);
	static bool handle_extension(const std::string &ext);
	static const bool has_transparency = true;
	static const bool is_lossless = true;

	static std::unique_ptr<Image<>> load(std::istream &in, 
											const parameters &p);

	static void save(std::ostream &out, const Image<> &img,
					 const parameters &p);
	static void save(std::ostream &out, Image<> &&img, const parameters &p);

	static auto create_large(const image_traits<> &traits, const r2::usize &s, 
							 const std::string &fname, const parameters &p)
		-> std::unique_ptr<large_image_impl>;

	static auto load_large(const std::string &fname, const parameters &p)
		-> std::unique_ptr<large_image_impl>;
};


}} // namespace s3d::img

#endif

// $Id: bitmap.h 2977 2010-08-18 22:37:40Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

