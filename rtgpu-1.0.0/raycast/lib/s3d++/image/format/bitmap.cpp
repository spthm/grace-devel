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

#include "../pch.h"
#include "bitmap.h"
#include "util.h"
#include "../../math/r2/size.h"
#include "../image.h"
#include "../../color/color.h"
#include "../../util/hash.h"
#include "../../util/pointer.h"
#include "../large_image.h"

namespace s3d { namespace img
{
	namespace
	{
		size_t calc_padding(size_t w, int bpp)/*{{{*/
		{
			size_t padding;
			switch(bpp)
			{
			case 4:
				padding = 0;
				break;
			case 3:
			case 1:
				padding = 4 - (w*bpp)%4;
				if(padding == 4)
					padding = 0;
				break;
			default:
				throw std::runtime_error("Invalid pixel type");
			}
			return padding;
		}/*}}}*/
	const image_traits<> &get_traits(size_t bpp)/*{{{*/
	{
		switch(bpp)
		{
		case 1:
			return y8::traits();
		case 3:
			return bgr8::traits();
		case 4:
			return bgra8::traits();
		default:
			assert(false);
			throw std::runtime_error("Invalid pixel type");
		}/*}}}*/
	}
}

class large_bitmap_impl : public large_image_impl/*{{{*/
{
public:
	large_bitmap_impl(int fd, const r2::usize &s, size_t offset, size_t bpp);

	~large_bitmap_impl();

private:
	int m_fd;
	size_t m_bpp;
	size_t m_offset;
	size_t m_stride;
		
	virtual std::unique_ptr<Image<>> do_read(const r2::ibox &bounds) const;
	virtual void do_write(const r2::ipoint &topleft, const Image<> &img,
						  const r2::ibox &bounds_img);
	virtual void do_write(const r2::ibox &bounds, const color::radiance &c);
};/*}}}*/

bool format_traits<BITMAP>::is_format(std::istream &in)/*{{{*/
{
    if(!in)
    	throw std::runtime_error("Bad stream state");

    int pos = in.tellg();

	bmp_header header;
	in.read((char *)&header, sizeof(header));
	if(!in || in.gcount() != sizeof(header))
	{
		in.seekg(pos, std::ios::beg);
		in.clear();
		return false;
	}

	bmp_dib_v3_header info;
	in.read((char *)&info, sizeof(info));
	if(!in || in.gcount() != sizeof(info))
	{
		in.seekg(pos, std::ios::beg);
		in.clear();
		return false;
	}

    in.seekg(pos, std::ios::beg);

	return header.type[0]=='B' && header.type[1]=='M' && 
		info.size == sizeof(info);
}/*}}}*/
bool format_traits<BITMAP>::handle_extension(const std::string &ext)/*{{{*/
{
	return boost::to_upper_copy(ext) == "BMP";
}/*}}}*/

std::unique_ptr<Image<>> format_traits<BITMAP>::load(std::istream &in,/*{{{*/
									const parameters &p)
{
	bmp_header header;
	in.read((char *)&header, sizeof(header));
	if(in.gcount() != sizeof(header))
		throw std::runtime_error("Error reading bitmap header");

	bmp_dib_v3_header info;
	in.read((char *)&info, sizeof(info));
	if(in.gcount() != sizeof(info))
		throw std::runtime_error("Error reading bitmap info");

	if(header.type[0]!='B' || header.type[1] != 'M')
		throw std::runtime_error("Bitmap type not supported");
	if(info.size != sizeof(info))
		throw std::runtime_error("Bitmap version not supported");
	if(info.compression != 0)
		throw std::runtime_error("Cannot handle compressed bitmaps");

	std::unique_ptr<Image<>> img;

	switch(info.bits)
	{
	case 8:
		img.reset(new y8(r2::isize(info.width, info.height)));
		break;
	case 24:
		img.reset(new bgr8(r2::isize(info.width, info.height)));
		break;
	case 32:
		img.reset(new bgra8(r2::isize(info.width, info.height)));
		break;
	default:
		throw std::runtime_error("Cannot handle non RGB(A)/grayscale bitmaps");
	}

	int padding = 4 - (info.width*info.bits/8)%4;
	if(padding == 4)
		padding = 0;

	// The bitmap y coordinate goes from bottom to top
	int curpos = (img->h()-1)*img->row_stride();

	for(int j=img->h()-1; j>=0; --j)
	{		
		in.read((char *)img->data()+curpos, img->w()*img->col_stride());
		if(in.gcount() != (int)(img->w()*img->col_stride()))
			throw std::runtime_error("Error reading bitmap data");

		if(padding != 0)
			in.seekg(padding, std::ios::cur);

		curpos -= img->row_stride();
	}

	return std::move(img);
}/*}}}*/

large_bitmap_impl::large_bitmap_impl(int fd, const r2::usize &s, size_t offset, size_t bpp)/*{{{*/
	: large_image_impl(s, get_traits(bpp))
	, m_fd(fd), m_bpp(bpp), m_offset(offset)
{
	m_stride = s.w*bpp + calc_padding(s.w, bpp);
}/*}}}*/
large_bitmap_impl::~large_bitmap_impl() /*{{{*/
{ 
	::close(m_fd); 
}/*}}}*/

auto large_bitmap_impl::do_read(const r2::ibox &bounds) const/*{{{*/
	-> std::unique_ptr<Image<>> 
{
	r2::ibox area = bounds & r2::ibox(0,0,w(), h());

	std::unique_ptr<Image<>> img = traits().create_image(area.size);

	if(area.is_zero())
		return std::move(img);

	off_t off = m_offset+(h()-area.y-area.h)*m_stride+area.x*3;

	if(::lseek(m_fd, off, SEEK_SET) == (off_t)-1)
		throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));

	off_t curpos = (area.h-1)*(off_t)img->row_stride();

	for(int j=0; j<area.h; ++j)
	{		
		ssize_t nread = ::read(m_fd, img->data()+curpos, area.w*m_bpp);		
		if(nread != ssize_t(area.w*m_bpp))
			throw std::runtime_error("Error reading bitmap");

		if(::lseek(m_fd, m_stride-area.w*3, SEEK_CUR) == (off_t)-1)
			throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));

		curpos -= img->row_stride();
	}

	return std::move(img);
}/*}}}*/
void large_bitmap_impl::do_write(const r2::ipoint &pos, const Image<> &_img,/*{{{*/
								 const r2::ibox &bounds_img)
{

	// shared pra poder especificar o deleter
	std::shared_ptr<const Image<>> img;

	switch(m_bpp)
	{
	case 1:
		img = image_cast_ptr<y8>(_img);
		break;
	case 3:
		img = image_cast_ptr<bgr8>(_img);
		break;
	case 4:
		img = image_cast_ptr<bgra8>(_img);
		break;
	default:
		assert(false);
		throw std::runtime_error("Invalid pixel type");
	}

	assert(img->col_stride() == m_bpp);

	r2::ibox dst_bounds(pos, bounds_img.size);
	dst_bounds &= r2::ibox(0,0,w(),h());

	size_t dw = dst_bounds.w,
		   dh = dst_bounds.h;

	int jbeg = max(0,bounds_img.y + pos.y - dst_bounds.y),
		jend = min(img->h(),jbeg + dh),
		ibeg = max(0,bounds_img.x + pos.x - dst_bounds.x);

	size_t col_stride = img->col_stride();

	off_t curpos = ibeg*col_stride + (jend-1)*img->row_stride();

	off_t off = m_offset+off_t(h()-dst_bounds.y-dh)*m_stride+dst_bounds.x*col_stride;

	if(::lseek(m_fd, off, SEEK_SET) == (off_t)-1)
		throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));

	for(int j=jend-1; j>=jbeg; --j)
	{
		ssize_t nwritten = ::write(m_fd, img->data()+curpos,dw*col_stride);
		if((size_t)nwritten != dw*col_stride)
			throw std::runtime_error("Error writing ");

		if(::lseek(m_fd, m_stride-dw*col_stride, SEEK_CUR) == (off_t)-1)
			throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));
		curpos -= img->row_stride();
	}
}/*}}}*/
void large_bitmap_impl::do_write(const r2::ibox &bounds, /*{{{*/
								 const color::radiance &c)
{
	r2::ibox area = bounds & r2::ibox(0,0,w(),h());

	off_t off = m_offset+(h()-area.y-area.h)*m_stride+area.x*m_bpp;

	if(::lseek(m_fd, off, SEEK_SET) == (off_t)-1)
		throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));

	std::vector<uint8_t> rowdata;

	switch(m_bpp)
	{
	case 1:
		rowdata.resize(area.w, color_cast<color::y8>(c));
		break;
	case 3:
	case 4:
		rowdata.reserve(area.w*m_bpp);
		{
			auto _c = color_cast<color::bgr8>(c);
			for(int i=0; i<area.w; ++i)
			{
				rowdata.push_back(_c.b);
				rowdata.push_back(_c.g);
				rowdata.push_back(_c.r);
				if(m_bpp==4)
					rowdata.push_back(255);
			}
		}
		break;
	default:
		assert(false);
	}

	for(int j=0; j<area.h; ++j)
	{
		int nwritten = ::write(m_fd, &rowdata[0], rowdata.size());
		if(nwritten != (int)rowdata.size())
			throw std::runtime_error("Error writing into bitmap");
		if(::lseek(m_fd, m_stride-rowdata.size(), SEEK_CUR) == (off_t)-1)
			throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));
	}
}/*}}}*/

auto format_traits<BITMAP>::load_large(const std::string &fname,/*{{{*/
									   const parameters &p)
	-> std::unique_ptr<large_image_impl> 
{
	int fd = ::open(fname.c_str(), O_RDWR|O_LARGEFILE/*|O_BINARY*/);
	if(fd < 0)
		throw std::runtime_error(fname+": "+strerror(errno));
	try
	{
		bmp_header header;
		int rcount = ::read(fd, &header, sizeof(header));
		if(rcount != sizeof(header))
			throw std::runtime_error("Error reading bitmap header");

		if(header.type[0]!='B' || header.type[1] != 'M')
			throw std::runtime_error("Bitmap type not supported");

		bmp_dib_v3_header info;
		rcount = ::read(fd, &info, sizeof(info));
		if(rcount != sizeof(info))
			throw std::runtime_error("Error reading bitmap header");

		if(info.size != sizeof(info))
			throw std::runtime_error("Bitmap version not supported");
		if(info.compression != 0)
			throw std::runtime_error("I cannot handle compressed bitmaps");

		return make_unique<large_bitmap_impl>
			(fd, r2::usize{info.width, info.height}, header.offset,info.bits/8);
	}
	catch(...)
	{
		::close(fd);
		throw;
	}
}/*}}}*/
auto format_traits<BITMAP>::create_large(const image_traits<> &traits, /*{{{*/
						 const r2::usize &s, const std::string &fname,
						 const parameters &p)
	-> std::unique_ptr<large_image_impl>
{
	// CUIDADO em outras plataformas, o arquivo tem que ser aberto em
	// modo BINÁRIO!
	int fd = ::open(fname.c_str(), 
		O_CREAT|O_RDWR|O_TRUNC|O_LARGEFILE/*|O_BINARY*/,
		0660
	);

	if(fd < 0)
		throw std::runtime_error(fname+": "+strerror(errno));

	size_t bpp;

	if(traits.has_alpha)
		bpp = 4;
	else if(traits.model == model::GRAYSCALE)
		bpp = 1;
	else
		bpp = 3;

	try
	{
		size_t padding = calc_padding(s.w, bpp);
		size_t row_stride = s.w*bpp + padding;

		bmp_header header;

		header.type[0] = 'B';
		header.type[1] = 'M';
		header.reserved1 = header.reserved2 = 0;
		header.offset = sizeof(bmp_header)+sizeof(bmp_dib_v3_header);

		if(bpp == 1)
			header.offset += 256*4; // accounts for palette for grayscale image

		header.size = header.offset + row_stride*s.h;
		int nwritten = ::write(fd, &header, sizeof(header));
		if(nwritten != sizeof(header))
			throw std::runtime_error("Error writing into "+fname);

		bmp_dib_v3_header info;
		info.size = sizeof(info);
		info.width = s.w;
		info.height = s.h;
		info.planes = 1;
		info.bits = bpp*8;
		info.compression = 0; // no compression
		info.imagesize = header.size-header.offset;
		info.xresolution = 1024; // arbitrario ?
		info.yresolution = 1024; 
		info.ncolours = 0;
		info.importantcolours = 0;
		nwritten = ::write(fd, &info, sizeof(info));
		if(nwritten != sizeof(info))
			throw std::runtime_error("Error writing into "+fname);

		// se for grayscale, temos que criar uma palette um uma rampa de cinza
		if(bpp == 1)
		{
			color::bgra8 palette[256];
			assert(sizeof(palette) == 256*4);
			for(int i=0; i<256; ++i)
				palette[i] = color::bgr8(i,i,i);

			if(::write(fd, &palette, sizeof(palette)) != sizeof(palette))
				throw std::runtime_error("Error writing bitmap palette");
		}

		// Vamos escrever um byte na última posição do arquivo p/ o tamanho dele
		// ficar correto.
		if(::lseek(fd, header.offset+s.h*row_stride-1, SEEK_SET) == (off_t)-1)
			throw std::runtime_error("Error seeking" + fname + strerror(errno));

		char c = 0;
		nwritten = ::write(fd, &c, 1);
		if(nwritten != 1)
			throw std::runtime_error("Error writing into "+fname);
		if(::lseek(fd, header.offset, SEEK_SET) == (off_t)-1)
			throw std::runtime_error("Error seeking" + fname + strerror(errno));

		return make_unique<large_bitmap_impl>(fd,s, header.offset, bpp);
	}
	catch(...)
	{
		::close(fd);
		unlink(fname.c_str());
		throw;
	}
}/*}}}*/

namespace
{
	void save_bmp(std::ostream &out, const Image<> &img,/*{{{*/
			  const parameters &p)
	{
		int padding = calc_padding(img.w(), img.col_stride());

		int row_stride = img.w()*img.col_stride() + padding;

		bmp_header header;
		header.type[0] = 'B';
		header.type[1] = 'M';
		header.reserved1 = header.reserved2 = 0;
		header.offset = sizeof(bmp_header)+sizeof(bmp_dib_v3_header);
		header.size = header.offset + row_stride*img.h();
		out.write((const char *)&header, sizeof(header));

		bmp_dib_v3_header info;
		info.size = sizeof(info);
		info.width = img.w();
		info.height = img.h();
		info.planes = 1;
		info.bits = img.col_stride()*8;
		info.compression = 0; // no compression
		info.imagesize = header.size-header.offset;
		info.xresolution = 1024; // arbitrario ?
		info.yresolution = 1024; 
		info.ncolours = 0;
		info.importantcolours = 0;
		out.write((const char *)&info, sizeof(info));

		assert(img.traits().depth == 8);
		assert(img.traits().dim == 3 || img.traits().dim==1 || 
			   img.traits().dim==4);
		assert(img.traits().invert);

		int curpos = (img.h()-1)*img.row_stride();

		for(int j=img.h(); j; --j)
		{
			const char *data = (const char *)(img.data()+curpos);

			// TODO: under certain conditions we might be reading 
			// 'padding' bytes after the end of img.data()
			out.write(data, img.w()*img.col_stride()+padding);

			curpos -= img.row_stride();
		}
	}/*}}}*/
template <class I>
void save_img(std::ostream &out, I &&img, const parameters &p)/*{{{*/
{
	if(img.traits().model == model::GRAYSCALE)
		save_bmp(out, image_cast<y8>(std::forward<I>(img)), p);
	else
	{
		if(img.traits().has_alpha)
			save_bmp(out, image_cast<bgra8>(std::forward<I>(img)), p);
		else
			save_bmp(out, image_cast<bgr8>(std::forward<I>(img)), p);
	}
}/*}}}*/

class save_visitor /*{{{*/
	: public const_visitor_impl<bgra8, bgr8, y8>
{
	struct impl
	{
		template <class T>
		bool operator()(std::ostream &out, const T &img, const parameters &p)
		{
			save_bmp(out, img, p);
			return true;
		}
	};

public:
	save_visitor(std::ostream &out, const parameters &p) 
	: visitor_type(std::bind<bool>(impl(), std::ref(out), std::placeholders::_1,
								 std::ref(p))) {}
};/*}}}*/
}

void format_traits<BITMAP>::save(std::ostream &out, const Image<> &img,/*{{{*/
							  const parameters &p)
{
	if(img.accept(save_visitor(out, p)))
		return;

	save_img(out, img, p);
}/*}}}*/
void format_traits<BITMAP>::save(std::ostream &out, Image<> &&img,/*{{{*/
							  const parameters &p)
{
	if(img.accept(save_visitor(out, p)))
		return;

	save_img(out, std::move(img), p);
}/*}}}*/

}} // namespace s3d::img

// $Id: bitmap.cpp 3135 2010-09-16 23:33:52Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

