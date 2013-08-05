#include "pch.h"
#include "large_image.h"
#include "traits.h"
#include "image.h"

namespace s3d { namespace img
{

large_image<>::large_image(std::unique_ptr<large_image_impl> _pimpl)
	: pimpl(std::move(_pimpl))
{
}

std::unique_ptr<Image<>> large_image<>::read(const r2::ibox &bounds) const
{ 
	return pimpl->read(bounds); 
}


std::unique_ptr<Image<>> large_image_impl::read(const r2::ibox &bounds) const
{ 
	return do_read(bounds); 
}

void large_image<>::write(const r2::ipoint &topleft, const Image<> &img,
						  const r2::ibox &bounds_img)
{ 
	pimpl->write(topleft, img, bounds_img & bounds(img)); 
}

void large_image<>::write(const r2::ibox &bounds, const color::radiance &c)
{ 
	pimpl->write(bounds, c); 
}

void large_image<>::write(const r2::ipoint &topleft, const Image<> &img)
{ 
	return write(topleft, img, bounds(img));
}

r2::ibox bounds(const large_image<> &img)
{
	return {0,0, img.w(), img.h()};
}


default_large_image_impl::default_large_image_impl(std::unique_ptr<Image<>> img,
						 const std::string &fname, const parameters &params,
						 bool is_new)
	: large_image_impl(img->size(), img->traits())
	, m_image(std::move(img))
	, m_dirty(is_new)
	, m_fname(fname)
	, m_params(params)
{
}

default_large_image_impl::~default_large_image_impl()
{
	if(m_dirty)
	{
		assert(m_image);
		img::save(m_fname, std::move(*m_image), m_params);
	}
}

auto default_large_image_impl::do_read(const r2::ibox &bounds) const
	-> std::unique_ptr<Image<>> 
{
	assert(m_image);
	return to_pointer(copy(*m_image, bounds));
}

void default_large_image_impl::do_write(const r2::ipoint &p, const Image<> &img,
										const r2::ibox &_bounds_img)
{
	auto bounds_img = _bounds_img & bounds(img);

	r2::ibox area = (bounds_img+(p-r2::iorigin)) & r2::ibox(0,0,w(),h());

	if(area.is_zero())
		return;

	copy_into(*m_image, area, img, bounds_img);

	m_dirty = true;
}

void default_large_image_impl::do_write(const r2::ibox &bounds, 
										const color::radiance &_c)
{
	r2::ibox area = bounds & r2::ibox(0,0,w(),h());

	if(area.is_zero())
		return;

	m_dirty = true;

	auto *pixline = m_image->pixel_ptr_at(area.origin);
	size_t stride = area.w*m_image->col_stride();

	auto c = m_image->traits().color_cast(_c);
	assert(c.size() == m_image->col_stride());
	assert(c.size() == m_image->traits().bpp);

	typedef std::not_equal_to<decltype(*pixline)> not_equal;

	// cor composta pela sequencia de bytes iguais,
	if(adjacent_find(c.begin(), c.end(), not_equal()) == c.end())
	{
		if((size_t)area.w != w())
		{
			for(int i=area.h; i; --i, pixline += m_image->row_stride())
				std::fill(pixline, pixline+stride, *c.begin());
		}
		else
		{
			assert(area.x == 0);
			std::fill(pixline, pixline+stride*area.h, *c.begin());
		}
	}
	else
	{
		if((size_t)area.w != w())
		{
			for(int i=area.h; i; --i, pixline += m_image->row_stride())
			{
				auto *pix = pixline;
				for(int j=area.w; j; --j, pix += m_image->col_stride())
					std::copy(c.begin(), c.end(), pix);
			}
		}
		else
		{
			assert(area.x == 0);
			auto *pix = pixline;
			for(int k=area.w*area.h; k; --k, pix += m_image->col_stride())
				std::copy(c.begin(), c.end(), pix);
		}
	}
}

}}
