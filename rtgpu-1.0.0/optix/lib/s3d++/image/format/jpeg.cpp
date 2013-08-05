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

// $Id: jpeg.cpp 2978 2010-08-19 02:03:55Z rodolfo $
// vim: nocp:ci:sts=4:fdm=marker:fmr={{{,}}}
// vi: ai:sw=4:ts=8

#include "../pch.h"
#include "../image.h"
#include "../params.h"
#include "../large_image.h"
#include "../../color/luminance.h"
#include "../../color/rgb.h"
#include "jpeg.h"

#if HAS_JPEG

namespace s3d { namespace img
{

namespace
{

	void myjpeg_error_exit(j_common_ptr cinfo)/*{{{*/
	{
		char buffer[JMSG_LENGTH_MAX];
		(*cinfo->err->format_message)(cinfo,buffer);
		longjmp(*reinterpret_cast<jmp_buf *>(cinfo->client_data), 1);
	}/*}}}*/

	struct my_source_mgr : jpeg_source_mgr/*{{{*/
	{
		my_source_mgr(std::istream &_in) : in(_in) {}
		std::istream &in;
		std::vector<JOCTET> buffer;
	};/*}}}*/
	void stream_init_source(j_decompress_ptr dinfo)/*{{{*/
	{
		my_source_mgr &src = static_cast<my_source_mgr &>(*dinfo->src);
		src.buffer.clear();
		src.next_input_byte = NULL;
		src.bytes_in_buffer = 0;
	}/*}}}*/
	boolean stream_fill_input_buffer(j_decompress_ptr dinfo)/*{{{*/
	{
		my_source_mgr &src = static_cast<my_source_mgr &>(*dinfo->src);

		src.buffer.resize(4096);
		src.in.read((char *)&src.buffer[0], src.buffer.size());

		size_t nbytes = src.in.gcount();

		if(nbytes <= 0)
		{
			src.buffer[0] = (JOCTET)0xFF;
			src.buffer[1] = (JOCTET)JPEG_EOI;
			nbytes = 2;
		}
		src.next_input_byte = &src.buffer[0];
		src.bytes_in_buffer = nbytes;
		return true;
	}/*}}}*/
	void stream_skip_input_data(j_decompress_ptr dinfo, long num_bytes)/*{{{*/
	{
		my_source_mgr &src = static_cast<my_source_mgr &>(*dinfo->src);

		if(num_bytes > 0)
		{
			if((long)src.bytes_in_buffer > num_bytes)
			{
				src.bytes_in_buffer -= num_bytes;
				src.next_input_byte += num_bytes;
			}
			else
			{
				src.in.seekg(num_bytes - src.bytes_in_buffer, std::ios::cur);
				src.bytes_in_buffer = 0;
				src.next_input_byte = NULL;
			}
		}
	}/*}}}*/
	boolean stream_resync_to_restart(j_decompress_ptr dinfo, int desired)/*{{{*/
	{
		my_source_mgr &src = static_cast<my_source_mgr &>(*dinfo->src);

		src.in.seekg(0, std::ios::beg);
		src.buffer.clear();
		src.next_input_byte = NULL;
		src.bytes_in_buffer = 0;
		return true;
	}/*}}}*/
	void stream_term_source(j_decompress_ptr dinfo)/*{{{*/
	{
		//my_source_mgr &src = static_cast<my_source_mgr &>(*dinfo->src);
	}/*}}}*/

	struct my_destination_mgr : jpeg_destination_mgr/*{{{*/
	{
		my_destination_mgr(std::ostream &_out) : out(_out) {}
		std::ostream &out;
		std::vector<JOCTET> buffer;
	};/*}}}*/
	void stream_init_destination(j_compress_ptr cinfo)/*{{{*/
	{
		my_destination_mgr &dest = static_cast<my_destination_mgr &>(*cinfo->dest);
		dest.buffer.resize(4096);
		dest.next_output_byte = &dest.buffer[0];
		dest.free_in_buffer = dest.buffer.size();
	}/*}}}*/
	boolean stream_empty_output_buffer(j_compress_ptr cinfo)/*{{{*/
	{
		my_destination_mgr &dest = static_cast<my_destination_mgr &>(*cinfo->dest);

		dest.out.write((char *)&dest.buffer[0], dest.buffer.size());

		dest.next_output_byte = &dest.buffer[0];
		dest.free_in_buffer = dest.buffer.size();

		return true;
	}/*}}}*/
	void stream_term_destination(j_compress_ptr cinfo)/*{{{*/
	{
		my_destination_mgr &dest = static_cast<my_destination_mgr &>(*cinfo->dest);

		dest.out.write((char *)&dest.buffer[0], dest.buffer.size()-dest.free_in_buffer);
	}/*}}}*/

	void save_jpeg(std::ostream &out, const Image<> &img, J_COLOR_SPACE color_space, int components, const parameters &p)/*{{{*/
	{
		jpeg_compress_struct cinfo;
		jpeg_error_mgr jerr;
		jpeg_create_compress(&cinfo);

		try
		{
			jmp_buf jump;
			cinfo.client_data = &jump;
			if(setjmp(jump))
				throw std::runtime_error("Error writing JPEG");

			cinfo.err = jpeg_std_error(&jerr);
			jerr.error_exit = myjpeg_error_exit;

			my_destination_mgr dmgr(out);
			dmgr.init_destination = &stream_init_destination;
			dmgr.empty_output_buffer = &stream_empty_output_buffer;
			dmgr.term_destination = &stream_term_destination;
			cinfo.dest = &dmgr;

			cinfo.image_width = img.w();
			cinfo.image_height = img.h();
			cinfo.input_components = components;
			cinfo.in_color_space = color_space;
			jpeg_set_defaults(&cinfo);

			if(p[quality])
			{
				float q = any_cast<float>(p[quality]);
				jpeg_set_quality(&cinfo, std::round(q), false);
			}

			jpeg_start_compress(&cinfo, true);

			JSAMPROW row[1];
			size_t row_stride = img.row_stride();
			while(cinfo.next_scanline < cinfo.image_height)
			{
				row[0] = (JSAMPLE *)(img.data() + cinfo.next_scanline*row_stride);
				if(jpeg_write_scanlines(&cinfo, row, 1) < 1)
					break;
			}

			jpeg_finish_compress(&cinfo);
			jpeg_destroy_compress(&cinfo);
		}
		catch(...)
		{
			jpeg_destroy_compress(&cinfo);
			throw;
		}
	}/*}}}*/

template <class I>
void save_img(std::ostream &out, I &&img, const parameters &p)/*{{{*/
{
	if(img.traits().model == model::GRAYSCALE)
		save_jpeg(out, image_cast<y8>(std::forward<I>(img)), JCS_GRAYSCALE, 1, p);
	else
		save_jpeg(out, image_cast<rgb8>(std::forward<I>(img)), JCS_RGB, 3, p);
}/*}}}*/

class save_visitor /*{{{*/
	: public const_visitor<rgb8, y8>
{
public:
	save_visitor(std::ostream &out, const parameters &p) 
		: m_out(out), m_params(p) {}

private:
	std::ostream &m_out;
	const parameters &m_params;

	virtual bool do_visit(const rgb8 &img)
	{
		save_jpeg(m_out, img, JCS_RGB, 3, m_params);
		return true;
	}
	virtual bool do_visit(const y8 &img)
	{
		save_jpeg(m_out, img, JCS_GRAYSCALE, 1, m_params);
		return true;
	}
};/*}}}*/

}

bool format_traits<JPEG>::is_format(std::istream &in)/*{{{*/
{
	if(!in)
		throw std::runtime_error("Bad stream state");

	int pos = in.tellg();

	jpeg_decompress_struct dinfo;
	jpeg_error_mgr jerr;
	jpeg_create_decompress(&dinfo);

	my_source_mgr smgr(in);
	smgr.init_source = &stream_init_source;
	smgr.fill_input_buffer = &stream_fill_input_buffer;
	smgr.skip_input_data = &stream_skip_input_data;
	smgr.resync_to_restart = &stream_resync_to_restart;
	smgr.term_source = &stream_term_source;
	dinfo.src = &smgr;

	jmp_buf jump;
	dinfo.client_data = &jump;
	if(setjmp(jump))
	{
		in.clear();
		in.seekg(pos, std::ios::beg);
		return false;
	}

	dinfo.err = jpeg_std_error(&jerr);
	jerr.error_exit = myjpeg_error_exit;

	bool valid = jpeg_read_header(&dinfo, true) == JPEG_HEADER_OK;

	if(!in)
		in.clear();

	in.seekg(pos, std::ios::beg);

	jpeg_destroy_decompress(&dinfo);
	return valid;
}/*}}}*/
bool format_traits<JPEG>::handle_extension(const std::string &ext)/*{{{*/
{
	std::string EXT = boost::to_upper_copy(ext);

	return EXT == "JPG" || EXT == "JPEG";
}/*}}}*/

std::unique_ptr<Image<>> format_traits<JPEG>::load(std::istream &in,/*{{{*/
													  const parameters &p)

{
	jpeg_decompress_struct dinfo;
	jpeg_error_mgr jerr;
	jpeg_create_decompress(&dinfo);

	try
	{
		jmp_buf jump;
		dinfo.client_data = &jump;
		if(setjmp(jump))
			throw std::runtime_error("Error reading JPEG");

		dinfo.err = jpeg_std_error(&jerr);
		jerr.error_exit = myjpeg_error_exit;

		my_source_mgr smgr(in);
		smgr.init_source = &stream_init_source;
		smgr.fill_input_buffer = &stream_fill_input_buffer;
		smgr.skip_input_data = &stream_skip_input_data;
		smgr.resync_to_restart = &stream_resync_to_restart;
		smgr.term_source = &stream_term_source;
		dinfo.src = &smgr;

		jpeg_read_header(&dinfo, true);

		if(p[scale])
		{
			using std::abs;
			using std::ceil;

			float s = any_cast<float>(p[scale]);
			if(s <= 0)
				throw std::invalid_argument("Invalid scale");

			int iscale = ceil(log2(s));

			if(iscale > 0)
			{
				dinfo.scale_num = 1<<iscale;
				dinfo.scale_denom = 1;
			}
			else
			{
				dinfo.scale_num = 1;
				dinfo.scale_denom = 1<<abs(iscale);
			}
		}

		if(p[grayscale] && any_cast<bool>(p[grayscale]) == true)
			dinfo.out_color_space = JCS_GRAYSCALE;

		jpeg_start_decompress(&dinfo);	

		std::unique_ptr<Image<>> img;
		switch(dinfo.out_color_components)
		{
		case 1:
			img.reset(new y8(r2::isize(dinfo.output_width, dinfo.output_height)));
			break;
		case 3:
			img.reset(new rgb8(r2::isize(dinfo.output_width, dinfo.output_height)));
			break;
		default:
			throw std::runtime_error("JPEG: Image type not supported");
		}

		// Segundo a documentacao, devemos respeitar o rec_outbuf_height
		// p/ termos uma descompressao eficiente

		// Caso mais comum
		if(dinfo.rec_outbuf_height == 1)
		{	
			size_t row_stride = img->byte_row_stride();

			uint8_t *d = img->data();
			while(dinfo.output_scanline < dinfo.output_height)
			{		
				jpeg_read_scanlines(&dinfo, (JSAMPARRAY)&d, 1);
				d += row_stride;
			}
		}
		else
		{
			size_t row_stride = img->row_stride();

			std::vector<uint8_t *> buffer;
			buffer.reserve(dinfo.rec_outbuf_height);
			for(int i=0; i<dinfo.rec_outbuf_height; ++i)
				buffer.push_back(img->data()+row_stride*i);

			while(dinfo.output_scanline < dinfo.output_height)
			{		
				size_t num_scanlines 
					= jpeg_read_scanlines(&dinfo, (JSAMPARRAY)&buffer[0],
										  dinfo.rec_outbuf_height);
				for(int ns=0; ns<dinfo.rec_outbuf_height; ++ns)
					buffer[ns] += row_stride*num_scanlines;
			}
		}
		jpeg_finish_decompress(&dinfo);
		jpeg_destroy_decompress(&dinfo);

		return std::move(img);
	}
	catch(...)
	{
		jpeg_destroy_decompress(&dinfo);
		throw;
	}
}/*}}}*/

void format_traits<JPEG>::save(std::ostream &out, const Image<> &img,/*{{{*/
							  const parameters &p)
{
	if(img.accept(save_visitor(out, p)))
		return;

	save_img(out, img, p);
}/*}}}*/
void format_traits<JPEG>::save(std::ostream &out, Image<> &&img,/*{{{*/
							  const parameters &p)
{
	if(img.accept(save_visitor(out, p)))
		return;

	save_img(out, std::move(img), p);
}/*}}}*/

auto format_traits<JPEG>::load_large(const std::string &fname, /*{{{*/
									 const parameters &p)
	-> std::unique_ptr<large_image_impl>
{
	std::ifstream in(fname);
	if(!in)
		throw std::runtime_error("Error opening "+fname);

	auto img = load(in, p);

	return make_unique<default_large_image_impl>(std::move(img), fname, p, 
												 false/*existing img*/);
}/*}}}*/
auto format_traits<JPEG>::create_large(const image_traits<> &traits, /*{{{*/
									  const r2::usize &s, 
									  const std::string &fname,
									  const parameters &p)
	-> std::unique_ptr<large_image_impl>
{
	return make_unique<default_large_image_impl>(traits.create_image(s), 
												 fname, p, true/*new img*/);
}/*}}}*/

}} // namespace s3d::img

#endif

// $Id: jpeg.cpp 2978 2010-08-19 02:03:55Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

