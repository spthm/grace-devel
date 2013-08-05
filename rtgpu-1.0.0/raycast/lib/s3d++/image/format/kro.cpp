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
#include "kro.h"
#include "util.h"
#include "../../color/luminance.h"
#include "../../color/rgb.h"
#include "../../util/endianess.h"
#include "../../math/r2/size.h"
#include "../image.h"
#include "../../util/hash.h"
#include "../../util/pointer.h"
#include "../large_image.h"

namespace s3d { namespace img
{

namespace
{
	const image_traits<> &get_traits(size_t depth, size_t ncomp)/*{{{*/
	{
		switch(depth)
		{
		case 8:
			if(ncomp == 3)
				return rgb8::traits();
			else
				return rgba8::traits();
			break;
		case 16:
			if(ncomp == 3)
				return rgb16::traits();
			else
				return rgba16::traits();
			break;
		case 32:
			if(ncomp == 3)
				return Image<color::RGB<float>>::traits();
			else
				return Image<color::alpha<color::RGB<float>,color::LAST>>::traits();
			break;
		default:
			assert(false);
			throw std::runtime_error("Bad pixel type");
		}
	}/*}}}*/
}

class large_kro_impl : public large_image_impl/*{{{*/
{
public:
	large_kro_impl(int fd, const r2::usize &s, size_t depth, size_t ncomp);

	~large_kro_impl();
private:
	int m_fd;
	size_t m_depth;
	size_t m_ncomp;
		
	virtual std::unique_ptr<Image<>> do_read(const r2::ibox &bounds) const;
	virtual void do_write(const r2::ipoint &topleft, const Image<> &img,
						  const r2::ibox &bounds_img);
	virtual void do_write(const r2::ibox &bounds, const color::radiance &c);
};/*}}}*/

bool format_traits<KRO>::is_format(std::istream &in)/*{{{*/
{
    if(!in)
    	throw std::runtime_error("Bad stream state");

    int pos = in.tellg();

	kro_header header;
	in.read((char *)&header, sizeof(header));
	if(!in || in.gcount() != sizeof(header))
	{
		in.seekg(pos, std::ios::beg);
		in.clear();
		return false;
	}

    in.seekg(pos, std::ios::beg);

	return strncmp(header.sig, "KRO", 3)==0 &&
		   (header.depth==8 || header.depth==16 || header.depth==32) &&
		   (header.ncomp==3 || header.ncomp == 4);
}/*}}}*/
bool format_traits<KRO>::handle_extension(const std::string &ext)/*{{{*/
{
	return boost::to_upper_copy(ext) == "KRO";
}/*}}}*/

std::unique_ptr<Image<>> format_traits<KRO>::load(std::istream &in,/*{{{*/
									const parameters &p)
{
	kro_header header;
	in.read((char *)&header, sizeof(header));
	if(in.gcount() != sizeof(header))
		throw std::runtime_error("Error reading kro header");

	endian_swap(header.width);
	endian_swap(header.height);
	endian_swap(header.depth);
	endian_swap(header.ncomp);

	if(strncmp(header.sig, "KRO", 3)!=0 ||
		   (header.depth!=8 && header.depth!=16 && header.depth!=32) ||
		   (header.ncomp!=3 && header.ncomp != 4))
	{
		throw std::runtime_error("Image isn't KRO");
	}

	std::unique_ptr<Image<>> img = get_traits(header.depth, header.ncomp).
									create_image({header.width,header.height});

	int curpos = 0;

	for(int j=img->h()-1; j>=0; --j)
	{		
		in.read((char *)img->data()+curpos, img->w()*img->col_stride());
		if(in.gcount() != (int)(img->w()*img->col_stride()))
			throw std::runtime_error("Error reading kro data");

		curpos += img->row_stride();
	}

	return std::move(img);
}/*}}}*/

large_kro_impl::large_kro_impl(int fd, const r2::usize &s, size_t depth, size_t ncomp)/*{{{*/
	: large_image_impl(s, get_traits(depth, ncomp))
	, m_fd(fd), m_depth(depth), m_ncomp(ncomp)
{
}/*}}}*/
large_kro_impl::~large_kro_impl() /*{{{*/
{ 
	::close(m_fd); 
}/*}}}*/

auto large_kro_impl::do_read(const r2::ibox &bounds) const/*{{{*/
	-> std::unique_ptr<Image<>> 
{
	r2::ibox area = bounds & r2::ibox(0,0,w(), h());

	std::unique_ptr<Image<>> img = traits().create_image(area.size);

	if(area.is_zero())
		return std::move(img);

	size_t bpp = m_ncomp*m_depth/8;


	off_t off = sizeof(kro_header)+(area.y*w()+area.x)*bpp;

	if(::lseek(m_fd, off, SEEK_SET) == (off_t)-1)
		throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));

	off_t curpos = 0;

	for(int j=0; j<area.h; ++j)
	{		
		ssize_t nread = ::read(m_fd, img->data()+curpos, area.w*bpp);		
		if(nread != ssize_t(area.w*bpp))
			throw std::runtime_error("Error reading kro");

		if(w()!=(size_t)area.w && ::lseek(m_fd, (w()-area.w)*bpp, SEEK_CUR)==(off_t)-1)
			throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));

		curpos += img->row_stride();
	}

	return std::move(img);
}/*}}}*/
void large_kro_impl::do_write(const r2::ipoint &pos, const Image<> &_img,/*{{{*/
							  const r2::ibox &bounds_img)
{
	// shared pra poder especificar o deleter
	std::shared_ptr<const Image<>> img; 	
	switch(m_depth)
	{
	case 8:
		if(m_ncomp == 3)
			img = image_cast_ptr<rgb8>(_img);
		else
			img = image_cast_ptr<rgba8>(_img);
		break;
	case 16:
		if(m_ncomp == 3)
			img = image_cast_ptr<rgb16>(_img);
		else
			img = image_cast_ptr<rgba16>(_img);
		break;
	case 32:
		if(m_ncomp == 3)
			img = image_cast_ptr<Image<color::RGB<float>>>(_img);
		else
			img = image_cast_ptr<Image<color::alpha<color::RGB<float>,color::LAST>>>(_img);
		break;
	}

	assert(img->col_stride() == m_ncomp*m_depth/8);

	r2::ibox dst_bounds(pos, bounds_img.size);
	dst_bounds &= r2::ibox(0,0,w(),h());

	size_t dw = dst_bounds.w,
		   dh = dst_bounds.h;

	int jbeg = max(0,bounds_img.y + pos.y - dst_bounds.y),
		jend = min(img->h(),jbeg + dh),
		ibeg = max(0,bounds_img.x + pos.x - dst_bounds.x);

	size_t col_stride = img->col_stride(),
		   row_stride = img->row_stride();

	off_t curpos = ibeg*col_stride + jbeg*row_stride;

	off_t off = sizeof(kro_header)+(off_t(dst_bounds.y*w()) + dst_bounds.x)*col_stride;

	if(::lseek(m_fd, off, SEEK_SET) == (off_t)-1)
		throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));

	for(int j=jend-1; j>=jbeg; --j)
	{
		ssize_t nwritten = ::write(m_fd, img->data()+curpos,dw*col_stride);
		if((size_t)nwritten != dw*col_stride)
			throw std::runtime_error("Error writing ");

		if(::lseek(m_fd, (w()-dw)*m_ncomp*m_depth/8, SEEK_CUR) == (off_t)-1)
			throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));
		curpos += row_stride;
	}
}/*}}}*/
void large_kro_impl::do_write(const r2::ibox &bounds, /*{{{*/
								 const color::radiance &c)
{
	r2::ibox area = bounds & r2::ibox(0,0,w(),h());

	off_t off = sizeof(kro_header)+(area.y*w()+area.x)*m_ncomp*m_depth/8;

	if(::lseek(m_fd, off, SEEK_SET) == (off_t)-1)
		throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));

	std::vector<uint8_t> rowdata;
	rowdata.reserve(area.w*m_ncomp*m_depth/8);

	switch(m_depth)
	{
	case 8:
		if(m_ncomp == 3)
		{
			auto _c = color_cast<color::rgb8>(c);
			for(int i=0; i<area.w; ++i)
				rowdata.insert(rowdata.end(), _c.begin(), _c.end());
		}
		else
		{
			assert(m_ncomp == 4);
			auto _c = color_cast<color::rgba8>(c);
			for(int i=0; i<area.w; ++i)
				rowdata.insert(rowdata.end(), _c.begin(), _c.end());
		}
		break;
	case 16:
		if(m_ncomp == 3)
		{
			auto _c = color_cast<color::rgb16>(c);
			for(int i=0; i<area.w; ++i)
				rowdata.insert(rowdata.end(), (uint8_t *)_c.begin(),
							                  (uint8_t *)_c.end());
		}
		else
		{
			auto _c = color_cast<color::rgba16>(c);
			for(int i=0; i<area.w; ++i)
				rowdata.insert(rowdata.end(), (uint8_t *)_c.begin(),
							                  (uint8_t *)_c.end());
		}
		break;
	case 32:
		if(m_ncomp == 3)
		{
			auto _c = color_cast<color::RGB<float>>(c);
			for(int i=0; i<area.w; ++i)
				rowdata.insert(rowdata.end(), (uint8_t *)_c.begin(),
							                  (uint8_t *)_c.end());
		}
		else
		{
			auto _c = color_cast<color::alpha<color::RGB<float>,color::LAST>>(c);
			for(int i=0; i<area.w; ++i)
				rowdata.insert(rowdata.end(), (uint8_t *)_c.begin(),
							                  (uint8_t *)_c.end());
		}
		break;
	}

	for(int j=0; j<area.h; ++j)
	{
		int nwritten = ::write(m_fd, &rowdata[0], rowdata.size());
		if(nwritten != (int)rowdata.size())
			throw std::runtime_error("Error writing into kro");
		if(::lseek(m_fd, w()*m_ncomp*m_depth/8-rowdata.size(), SEEK_CUR) == (off_t)-1)
			throw std::runtime_error(std::string("Error seeking: ") + strerror(errno));
	}
}/*}}}*/

auto format_traits<KRO>::load_large(const std::string &fname,/*{{{*/
									const parameters &p)
	-> std::unique_ptr<large_image_impl> 
{
	int fd = ::open(fname.c_str(), O_RDWR|O_LARGEFILE/*|O_BINARY*/);
	if(fd < 0)
		throw std::runtime_error(fname+": "+strerror(errno));
	try
	{
		kro_header header;
		int rcount = ::read(fd, &header, sizeof(header));
		if(rcount != sizeof(header))
			throw std::runtime_error("Error reading kro header");

		endian_swap(header.width);
		endian_swap(header.height);
		endian_swap(header.depth);
		endian_swap(header.ncomp);

		if(strncmp(header.sig, "KRO", 3)!=0 ||
			   (header.depth!=8 && header.depth!=16 && header.depth!=32) ||
			   (header.ncomp!=3 && header.ncomp != 4))
		{
			throw std::runtime_error("Image isn't KRO");
		}

		return make_unique<large_kro_impl>
			(fd, r2::usize{header.width, header.height}, header.depth, header.ncomp);
	}
	catch(...)
	{
		::close(fd);
		throw;
	}
}/*}}}*/
auto format_traits<KRO>::create_large(const image_traits<> &traits, /*{{{*/
									  const r2::usize &s,
									  const std::string &fname,
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

	size_t ncomp = traits.has_alpha ? 4 : 3;
	size_t depth;

	if(traits.is_integral)
	{
		if(traits.depth == 8)
			depth = 8;
		else
			depth = 16;
	}
	else
		depth = 32;

	try
	{
		kro_header header;
		strncpy(header.sig, "KRO", 3);
		header.version = 1;
		header.width = s.w;
		header.height = s.h;
		header.depth = depth;
		header.ncomp = ncomp;

		endian_swap(header.width);
		endian_swap(header.height);
		endian_swap(header.depth);
		endian_swap(header.ncomp);

		ssize_t nwritten = ::write(fd, &header, sizeof(header));
		if(nwritten != sizeof(header))
			throw std::runtime_error("Error writing into "+fname);

		endian_swap(header.width);
		endian_swap(header.height);
		endian_swap(header.depth);
		endian_swap(header.ncomp);

		// Vamos escrever um byte na última posição do arquivo p/ o tamanho dele
		// ficar correto.
		if(::lseek(fd, sizeof(kro_header)+s.h*header.depth/8*header.ncomp-1, SEEK_SET) == (off_t)-1)
			throw std::runtime_error("Error seeking " + fname + strerror(errno));

		char c = 0;
		nwritten = ::write(fd, &c, 1);
		if(nwritten != 1)
			throw std::runtime_error("Error writing into "+fname);
		if(::lseek(fd, sizeof(kro_header), SEEK_SET) == (off_t)-1)
			throw std::runtime_error("Error seeking " + fname + strerror(errno));

		return make_unique<large_kro_impl>(fd,s,header.depth,header.ncomp);
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
	void save_kro(std::ostream &out, const Image<> &img, size_t depth,/*{{{*/
			  size_t ncomp, const parameters &p)
	{
		kro_header header;
		strncpy(header.sig, "KRO", 3);
		header.version = 1;
		header.width = img.w();
		header.height = img.h();
		header.depth = depth;
		header.ncomp = ncomp;

		assert(depth == img.traits().depth);
		assert(ncomp == img.traits().dim);

		endian_swap(header.width);
		endian_swap(header.height);
		endian_swap(header.depth);
		endian_swap(header.ncomp);

		out.write((const char *)&header, sizeof(header));

		endian_swap(header.width);
		endian_swap(header.height);
		endian_swap(header.depth);
		endian_swap(header.ncomp);

		assert(img.col_stride() == ncomp*depth/8);

		int curpos = 0;

		for(int j=img.h(); j; --j)
		{
			out.write((const char *)img.data()+curpos,img.w()*img.col_stride());
			curpos += img.row_stride();
		}

	}/*}}}*/

template <class I>
void save_img(std::ostream &out, I &&img, const parameters &p)/*{{{*/
{
	if(img.traits().is_integral)
	{
		if(img.traits().depth == 8)
		{
			if(img.traits().has_alpha)
				save_kro(out, image_cast<rgba8>(std::forward<I>(img)), 8, 4, p);
			else
				save_kro(out, image_cast<rgb8>(std::forward<I>(img)), 8, 3, p);
		}
		else
		{
			if(img.traits().has_alpha)
				save_kro(out, image_cast<rgba16>(std::forward<I>(img)), 16, 4, p);
			else
				save_kro(out, image_cast<rgb16>(std::forward<I>(img)), 16, 3, p);
		}
	}
	else
	{
		if(img.traits().has_alpha)
			save_kro(out, image_cast<rgba>(std::forward<I>(img)), 32, 4, p);
		else
			save_kro(out, image_cast<rgb>(std::forward<I>(img)), 32, 3, p);
	}
}/*}}}*/

	class save_visitor /*{{{*/
		: public const_visitor_impl<rgb8, rgba8, rgb16, rgba16,
						Image<color::RGB<float>>, 
						Image<color::alpha<color::RGB<float>,color::LAST>>>
	{
		struct impl
		{
			template <class T>
			bool operator()(std::ostream &out, const T &img, const parameters &p)
			{
				save_kro(out, img, traits::depth<T>::value, 
								   traits::dim<T>::value, p);
				return true;
			}
		};

	public:
		save_visitor(std::ostream &out, const parameters &p) 
			: visitor_type(std::bind<bool>(impl(), 
						std::ref(out), std::placeholders::_1, std::ref(p))) {}
	};/*}}}*/
}

void format_traits<KRO>::save(std::ostream &out, const Image<> &img,/*{{{*/
							  const parameters &p)
{
	if(img.accept(save_visitor(out, p)))
		return;

	save_img(out, img, p);
}/*}}}*/
void format_traits<KRO>::save(std::ostream &out, Image<> &&img,/*{{{*/
							  const parameters &p)
{
	if(img.accept(save_visitor(out, p)))
		return;

	save_img(out, std::move(img), p);
}/*}}}*/

}} // namespace s3d::img

// $Id: kro.cpp 2896 2010-08-01 01:30:44Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

