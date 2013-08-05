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

// $Id: image.cpp 2974 2010-08-18 16:02:48Z rodolfo $
// vim: nocp:ci:sts=4:fdm=marker:fmr={{{,}}}
// vi: ai:sw=4:ts=8

#include "pch.h"
#include "image.h"
#include "../color/radiance.h"
#include <algorithm>
#include <numeric>
#include "../math/r3/vector.h"
#include "convolution.h"
#include "kernel.h"

namespace s3d { namespace img
{

image apply_gamma(const image &orig, real g)/*{{{*/
{
	image img(orig.size());
	apply_gamma_into(img, g);
	return std::move(orig);
}/*}}}*/
void apply_gamma_into(image &orig, real g)/*{{{*/
{
	using std::pow;

	BOOST_FOREACH(auto &r, orig)
	{
		r.x = pow(r.x, g);
		r.y = pow(r.y, g);
		r.z = pow(r.z, g);
	}
}/*}}}*/

image compose(const luminance &x, const luminance &y, const luminance &z)/*{{{*/
{
	return compose<color::radiance>(x, y, z);
}/*}}}*/

image gaussian_blur(const image &img, float sigma, int dim)/*{{{*/
{
	img::luminance r,g,b;
	decompose(img, r, g, b);

	auto mask = gaussian_kernel(sigma, dim);

	return compose(convolve(r,mask), convolve(g,mask), convolve(b,mask));
}/*}}}*/

img::image Image<>::to_radiance() const/*{{{*/
{ 
	return do_conv_to_radiance(); 
}/*}}}*/
img::timage Image<>::to_radiance_alpha() const/*{{{*/
{ 
	return do_conv_to_radiance_alpha(); 
}/*}}}*/

img::image Image<>::move_to_radiance()/*{{{*/
{ 
	return do_move_to_radiance(); 
}/*}}}*/
img::timage Image<>::move_to_radiance_alpha()/*{{{*/
{ 
	return do_move_to_radiance_alpha(); 
}/*}}}*/

void copy_into(Image<> &dest, const r2::ibox &_rcdest, /*{{{*/
			   const Image<> &orig, const r2::ibox &_rcorig)
{
	r2::ibox rcdest = _rcdest & r2::ibox(r2::iorigin, dest.size()),
			 rcorig = _rcorig & r2::ibox(r2::iorigin, orig.size());

	if(rcdest.is_zero() || rcorig.is_zero())
		return;

	std::shared_ptr<const Image<>> porig;
	if(typeid(dest) == typeid(orig))
		porig.reset(&orig, null_deleter);
	else
		porig = dest.traits().image_cast(orig);

	rcdest &= r2::ibox(rcdest.x, rcdest.y, rcorig.w, rcorig.h);

	const auto *oline = porig->pixel_ptr_at(rcorig.origin);
	auto       *dline = dest.pixel_ptr_at(rcdest.origin);

	assert(porig->col_stride() == dest.col_stride());

	size_t stride = porig->col_stride()*rcdest.w;

	for(int i=rcdest.h; i; --i)
	{
		std::copy(oline, oline+stride, dline);

		oline += porig->row_stride();
		dline += dest.row_stride();
	}
}/*}}}*/

auto copy(const Image<> &orig, const r2::ibox &bxorig)
	-> std::unique_ptr<Image<>>
{
	auto dest = orig.create(bxorig.size);
	copy_into(*dest, orig, bxorig);
	return std::move(dest);
}

}} // namespace s3d

// $Id: image.cpp 2974 2010-08-18 16:02:48Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

