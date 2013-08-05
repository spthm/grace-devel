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

#include <stdexcept>
#include <boost/algorithm/minmax_element.hpp>
#include "../math/interpol.h"
#include "../color/cast.h"
#include "../color/radiance.h"

namespace s3d { namespace img
{

// Image<> members --------------------------

namespace detail_img/*{{{*/
{
	void rescale_into(image &dest, const r2::ibox &bxdest,
			         const image &orig, const r2::ibox &bxorig);
	void copy_into(image &dest, const r2::ibox &bxdest, 
				   const image &orig, const r2::ibox &bxorig);
}/*}}}*/

inline Image<>::Image(void *data, size_t capacity, const math::r2::usize &sz,/*{{{*/
			   const image_traits<> &traits, size_t row_stride)
	: m_data(data)
	, m_capacity(capacity)
    , m_size(sz)
	, m_row_stride(row_stride ? row_stride : sz.w*traits.bpp)
	, m_traits(&traits)
{
}/*}}}*/

inline Image<>::Image(Image<> &&that)/*{{{*/
	: m_data(that.m_data)
	, m_capacity(that.m_capacity)
	, m_size(that.m_size)
	, m_row_stride(that.m_row_stride)
	, m_traits(that.m_traits)
{
	that.m_data = NULL;
	that.m_capacity = 0;
	that.m_size = {0,0};
	that.m_row_stride = 0;
}/*}}}*/

inline auto Image<>::move_assign_impl(Image &&that) -> Image &/*{{{*/
{
	if(this == &that)
		return *this;

	m_data = that.m_data;
	m_capacity = that.m_capacity;
	m_size = that.m_size;
	m_row_stride = that.m_row_stride;

	// traits must not be copied, we already have the correct traits
//	m_traits = that.m_traits;

	that.m_data = NULL;
	that.m_capacity = 0;
	that.m_size = {0,0};
	that.m_row_stride = 0;
	return *this;
}/*}}}*/
inline auto Image<>::copy_assign_impl(const Image &that) -> Image &/*{{{*/
{
	if(this == &that)
		return *this;

	m_data = that.m_data;
	m_capacity = that.m_capacity;
	m_size = that.m_size;
	m_row_stride = that.m_row_stride;
	return *this;
}/*}}}*/

inline auto Image<>::operator=(Image &&that) -> Image &/*{{{*/
{
	if(typeid(*this) != typeid(that))
		throw std::runtime_error("Unrelated image types during move assignment");
	return move_assign_impl(std::move(that));
}/*}}}*/

inline uint8_t *Image<>::pixel_ptr_at(const r2::ipoint &pt)/*{{{*/
{
	assert(region().contains(pt));
	return data() + pt.x*col_stride() + pt.y*row_stride();
}/*}}}*/
inline const uint8_t *Image<>::pixel_ptr_at(const r2::ipoint &pt) const/*{{{*/
{
	assert(r2::ibox(0, 0, w(), h()).contains(pt));
	return data() + pt.x*col_stride() + pt.y*row_stride();
}/*}}}*/

template<class P> 
bool Image<>::has_pixel_type() const/*{{{*/
{
	return dynamic_cast<const Image<P> *>(this) ? true : false;
}/*}}}*/

// Image<P> members -----------------------

template <class P> 
Image<P>::Image() /*{{{*/
	: Image<>(NULL, 0, {0,0}, traits())
	, m_row_stride(0)
{
}/*}}}*/

template <class P> 
Image<P>::Image(std::unique_ptr<P[]> data, const r2::usize &sz, size_t row_stride) /*{{{*/
	: Image<>(data.release(), sizeof(P[sz.h*(row_stride?row_stride:sz.w)]),
			  traits(), row_stride?row_stride:sz.w*sizeof(P))
	, m_row_stride(row_stride?row_stride:sz.w)
{
	assert(Image<>::length() == m_capacity);
}/*}}}*/
template <class P> 
Image<P>::Image(const r2::usize &sz, size_t row_stride) /*{{{*/
	: Image<>(new P[sz.h*(row_stride?row_stride:sz.w)], 
			  sizeof(P[sz.h*(row_stride?row_stride:sz.w)]),
			  sz, traits(), (row_stride?row_stride:sz.w)*sizeof(P))
	, m_row_stride(row_stride?row_stride:sz.w)
{
	assert(Image<>::length() == m_capacity);
}/*}}}*/
template <class P> 
Image<P>::Image(size_t w, size_t h, size_t row_stride) /*{{{*/
	: Image<>(new P[h*(row_stride?row_stride:w)], 
			  sizeof(P[h*(row_stride?row_stride:w)]),
			  {w,h}, traits(), (row_stride?row_stride:w)*sizeof(P))
	, m_row_stride(row_stride?row_stride:w)
{
	assert(Image<>::length() == m_capacity);
}/*}}}*/
template <class P> 
Image<P>::Image(const Image &that) /*{{{*/
	: Image<>(new P[that.length()], sizeof(P[that.length()]),
			  that.size(), traits(), that.byte_row_stride())
	, m_row_stride(that.m_row_stride)
{
	assert(Image<>::length() == m_capacity);
	assert(length() == that.length());

	std::copy(that.data(), that.data()+that.length(), data());
}/*}}}*/
template <class P> 
Image<P>::Image(Image &&that) /*{{{*/
	: Image<>(std::move(that))
	, m_row_stride(that.m_row_stride)
{
	that.m_row_stride = 0;
}/*}}}*/

template <class P> template <class T>
Image<P>::Image(const Image<T> &that, /*{{{*/
    	  typename std::enable_if<	
			!std::is_same<P,T>::value && std::is_convertible<T,P>::value &&
			(std::is_same<T,color::radiance>::value ||
			 std::is_same<T,color::radiance_alpha>::value ||
			 std::is_same<P,color::radiance>::value ||
			 std::is_same<P,color::radiance_alpha>::value)>::type*)
	: Image<>(new P[that.length()], sizeof(P[that.length()]),
			  that.size(), traits())
	, m_row_stride(that.row_stride())
{
	// do a normal copy, no move this time

	assert(Image<>::length() == m_capacity);

	auto itsrc = that.begin();
	auto itdst = begin();

	assert(length() == that.length());

	while(itsrc != that.end())
		*itdst++ = color_cast<P>(*itsrc++);
}/*}}}*/

template <class P> template <class T>
Image<P>::Image(Image<T> &&that, /*{{{*/
    	  typename std::enable_if<	
			!std::is_same<P,T>::value && std::is_convertible<T,P>::value &&
			(std::is_same<T,color::radiance>::value ||
			 std::is_same<T,color::radiance_alpha>::value ||
			 std::is_same<P,color::radiance>::value ||
			 std::is_same<P,color::radiance_alpha>::value)>::type*)
	: Image<>(NULL, 0, {0,0}, traits())
	, m_row_stride(0)
{
	move_ctor(std::move(that));
}/*}}}*/

template <class P> template <class C>
void Image<P>::clear(const C &color)/*{{{*/
{
	std::fill(begin(), end(), color_cast<P>(color));
}/*}}}*/

namespace detail_img/*{{{*/
{
	template <class FROM, class TO>
	void safe_assign(FROM *from, TO *to)
	{
		if(std::has_trivial_destructor<FROM>::value && 
		   std::has_trivial_copy_constructor<TO>::value)
		{
			*to = color_cast<TO>(*from);
		}
		else
		{
			if(!std::has_trivial_destructor<FROM>::value)
			{
				TO aux = color_cast<TO>(*from);
				from->~FROM();
				if(std::has_trivial_copy_constructor<TO>::value)
					*to = aux;
				else
					new(to) TO(aux);
			}
			else if(!std::has_trivial_copy_constructor<TO>::value)
				new(to) TO(color_cast<TO>(*from));
			else
				*to = color_cast<TO>(*from);
		}
	}
}/*}}}*/

template <class P> template <class T>
void Image<P>::move_ctor(Image<T> &&that, /*{{{*/
	  typename std::enable_if<!std::is_same<void,P>::value && 
					          sizeof(P) <= sizeof(T)>::type*)
{
	static_assert(is_colorspace<P>::value, "P must be a colorspace");
	static_assert(is_colorspace<T>::value, "T must be a colorspace");

	assert(data() == NULL);
	assert(length() == 0);

#ifndef NDEBUG
	size_t origlen = that.length();
#endif

	move_assign_impl(std::move(that));
	m_row_stride = that.m_row_stride;
	Image<>::m_row_stride = m_row_stride*sizeof(P);
	that.m_row_stride = 0;

	assert(Image<>::length() <= m_capacity);

	try
	{
		if(sizeof(P) == sizeof(T))
		{
			union conv_t
			{
				P *out;
				T *in;
			} conv = {data()};

			size_t len = length();
			assert(origlen == len);

			for(size_t i=0; i<len; ++i, ++conv.in)
				detail_img::safe_assign(conv.in, conv.out);
		}
		else
		{
			T *in = reinterpret_cast<T *>(data());
			P *out = data();

			size_t len = length();
			assert(origlen >= len);

			for(size_t i=0; i<len; ++i)
				detail_img::safe_assign(in++, out++);
		}
	}
	catch(std::bad_alloc &e)
	{
		// allocation might have failed because destination alignment
		// isn't supported P (v4f for instance)

		std::unique_ptr<T[]> origdata(reinterpret_cast<T *>(m_data));

		m_data = new P[length()];
		m_capacity = sizeof(P[length()]);

		auto itsrc = origdata.get();
		auto itdst = begin();

		assert(length() == origlen);

		while(itdst != end())
			*itdst++ = color_cast<P>(*itsrc++);
	}
}/*}}}*/
template <class P> template <class T>
void Image<P>::move_ctor(Image<T> &&that, /*{{{*/
	  typename std::enable_if<!std::is_same<void,P>::value && 
							   (sizeof(P) > sizeof(T))>::type*)
{
	static_assert(is_colorspace<P>::value, "P must be a colorspace");
	static_assert(is_colorspace<T>::value, "T must be a colorspace");

	if(that.length() == 0)
		return;

	assert(data() == NULL);
	assert(length() == 0);

#ifndef NDEBUG
	auto oldlength = that.length();
#endif

	std::unique_ptr<T[]> olddata(that.data());
	that.m_data = NULL; // so it won't be deleted in next line
	move_assign_impl(std::move(that));
	m_row_stride = that.m_row_stride;
	that.m_row_stride = 0;

	Image<>::m_row_stride = m_row_stride*sizeof(P);

	m_data = new P[length()];
	m_capacity = sizeof(P[length()]);

	auto itsrc = olddata.get();
	auto itdst = begin();

	assert(length() == oldlength);

	while(itdst != end())
		*itdst++ = color_cast<P>(*itsrc++);
}/*}}}*/

template <class P> 
Image<P> &Image<P>::operator=(const Image &that)/*{{{*/
{
	delete[] data();

	copy_assign_impl(that);

	m_row_stride = that.m_row_stride;

	m_data = new P[length()];
	m_capacity = sizeof(P[length()]);
	assert(Image<>::length() == m_capacity);

	std::copy(that.data(), that.data()+that.length(), data());
	return *this;
}/*}}}*/
template <class P> 
Image<P> &Image<P>::operator=(Image &&that)/*{{{*/
{
	delete[] data();

	move_assign_impl(std::move(that));
	m_row_stride = that.m_row_stride;
	that.m_row_stride = 0;
	return *this;
}/*}}}*/

template <class P> 
P &Image<P>::pixel_at(const r2::ipoint &pt)/*{{{*/
{
	return *pixel_ptr_at(pt);
}/*}}}*/
template <class P> 
const P &Image<P>::pixel_at(const r2::ipoint &pt) const/*{{{*/
{
	return *pixel_ptr_at(pt);
}/*}}}*/

template <class P> 
P *Image<P>::pixel_ptr_at(const r2::ipoint &pt)/*{{{*/
{
	assert(region().contains(pt));
	return data() + pt.x*col_stride() + pt.y*row_stride();
}/*}}}*/
template <class P> 
const P *Image<P>::pixel_ptr_at(const r2::ipoint &pt) const/*{{{*/
{
	assert(r2::ibox(0, 0, w(), h()).contains(pt));
	return data() + pt.x*col_stride() + pt.y*row_stride();
}/*}}}*/

template <class P>
bool Image<P>::do_accept(visitor_base &&visitor)/*{{{*/
{
	return visitor.visit(*this);
}/*}}}*/
template <class P>
bool Image<P>::do_move_accept(visitor_base &&visitor)/*{{{*/
{
	return visitor.visit(std::move(*this));
}/*}}}*/
template <class P>
bool Image<P>::do_accept(visitor_base &&visitor) const/*{{{*/
{
	return visitor.visit(*this);
}/*}}}*/

// free functions ----------------------------

template <class P> 
P bicubic_pixel(const Image<P> &img, const r2::point &p)/*{{{*/
{
	// Who said there's no local functions in C/C++?
	struct aux
	{
		static real cubic_weight(real x)
		{
			real r = 0;

			real a = x+2;
			if(a > 0)
				r += a*a*a;

			a = x+1;
			if(a > 0)
				r -= 4*a*a*a;

			if(x > 0)
				r += 6*x*x*x;

			a = x-1;
			if(a > 0)
				r -= 4*a*a*a;

			return r;
		}
	};

	r2::ipoint ip = floor(p);

	assert(img.region().contains(ip));

	r2::vector dp = p - ip;

	P c = color_cast<P>(color::radiance(0,0,0));

	real wx[4];
	for(int n=-1; n<=2; ++n)
		wx[n+1] = aux::cubic_weight(dp.x-n);

	for(int m=-1; m<=2; ++m)
	{
		using math::clamp;

		int py = clamp(m+ip.y, 0, img.h()-1);

		real wy = aux::cubic_weight(m-dp.y);

		for(int n=-1; n<=2; ++n)
		{
			int px = clamp(n+ip.x, 0, img.w()-1);

			c += img.pixel_at({px, py})*(wy*wx[n+1]);
		}
	}
	
	return c/36.0;
}/*}}}*/

template <class P> 
P bilinear_pixel(const Image<P> &img, const r2::point &p)/*{{{*/
{
	r2::ipoint ip = floor(p);

	assert(img.region().contains(ip));

	const P *top_left = &img.pixel_at(ip),
		    *top_right = top_left,
		    *bot_left = top_left,
		    *bot_right = top_left;

	if(ip.y+1 < (int)img.h())
		bot_right = (bot_left += img.row_stride());

	if(ip.x+1 < (int)img.w())
	{
		top_right += img.col_stride();
		bot_right += img.col_stride();
	}

	auto dp = p-ip;

	return math::bilerp(dp.x, dp.y, *top_left, *bot_left, 
					    dp.y, *top_right, *bot_right);
}/*}}}*/

template <class P>
r2::point centroid(const Image<P> &img)/*{{{*/
{
	return { img.w() / real(2.0), img.h() / real(2.0) };
}/*}}}*/

template <class P>
auto gradient_pixel(const Image<P> &img, const r2::ipoint &pos)/*{{{*/
	-> Vector<typename difference_type<P>::type,2> 
{
	auto *pix = &img.pixel_at(pos);

	Vector<typename difference_type<P>::type,2> grad;

	const int dr = img.row_stride(),
		      dc = img.col_stride();

	/*
	if(pos.x == 0)
		grad.x = 2*(pix[dc] - pix[0]);
	else if(pos.x == img.w()-1)
		grad.x = 2*(pix[0] - pix[-dc]);
	else
		grad.x = pix[dc] - pix[-dc];

	if(pos.y == 0)
		grad.y = 2*(pix[0] - pix[dr]);
	else if(pos.y == img.h()-1)
		grad.y = 2*(pix[-dr] - pix[0]);
	else
		grad.y = pix[-dr] - pix[dr];
		*/
	if(pos.x == 0)
		grad.x = 2*(pix[dc] - pix[0]);
	else if(pos.x == img.w()-1)
		grad.x = 2*(pix[0] - pix[-dc]);
	else
		grad.x = pix[dc] - pix[-dc];

	if(pos.y == 0)
		grad.y = 2*(pix[dr] - pix[0]);
	else if(pos.y == img.h()-1)
		grad.y = 2*(pix[0] - pix[-dr]);
	else
		grad.y = pix[dr] - pix[-dr];

	return grad;
}/*}}}*/

template <class P>
r2::ibox bounds(const Image<P> &img)/*{{{*/
{
	return { 0, 0, img.w(), img.h() };
}/*}}}*/

template <class P> 
real aspect(const Image<P> &img)/*{{{*/
{
	return real(img.w()) / img.h();
}/*}}}*/

template <class P>
Image<P> copy(const Image<P> &orig, const r2::ibox &bxorig)/*{{{*/
{
	Image<P> img(bxorig.size);
	copy_into(img, orig, bxorig);
	return std::move(img);
}/*}}}*/

template <class P>
void copy_into(Image<P> &dest, const image &orig)/*{{{*/
{
	copy_into(dest, r2::ibox(r2::iorigin, dest.size()),
			  orig, r2::ibox(r2::iorigin, orig.size()));
}/*}}}*/
template <class P>
void copy_into(Image<P> &dest, const r2::ibox &bxdest, const Image<P> &orig)/*{{{*/
{
	copy_into(dest, bxdest, orig, r2::ibox(r2::iorigin, orig.size()));
}/*}}}*/
template <class P>
void copy_into(Image<P> &dest, const Image<P> &orig, const r2::ibox &bxorig)/*{{{*/
{
	copy_into(dest, r2::ibox(r2::iorigin, dest.size()), orig, bxorig);
}/*}}}*/
template <class P>
void copy_into(Image<P> &dest, const r2::ibox &_rcdest, /*{{{*/
			   const Image<P> &orig, const r2::ibox &_rcorig)
{
	r2::ibox rcdest = _rcdest & r2::ibox(r2::iorigin, dest.size()),
			 rcorig = _rcorig & r2::ibox(r2::iorigin, orig.size());

	if(typeid(dest) != typeid(orig))
		throw std::runtime_error("Image type mismatch during copy_into");

	if(rcdest.is_zero() || rcorig.is_zero())
		return;

	rcdest &= r2::ibox(rcdest.x, rcdest.y, rcorig.w, rcorig.h);

	const auto *oline = orig.pixel_ptr_at(rcorig.origin);
	auto       *dline = dest.pixel_ptr_at(rcdest.origin);

	assert(orig.col_stride() == dest.col_stride());

	for(int i=rcdest.h; i; --i)
	{
		std::copy(oline, oline+orig.col_stride()*rcdest.w, dline);

		oline += orig.row_stride();
		dline += dest.row_stride();
	}
}/*}}}*/

template <class P>
Image<P> &flip_vert_inplace(Image<P> &img)/*{{{*/
{
	using std::swap_ranges;

	auto *orig = img.pixel_ptr_at({0,0}),
	     *dest = img.pixel_ptr_at({0,img.h()-1});

	auto *end = img.pixel_ptr_at({0,(int)ceil(img.h()/2)});

	while(orig != end)
	{
		auto *next_orig = orig+img.row_stride();

		swap_ranges(orig, next_orig, dest);
		orig = next_orig;
		dest -= img.row_stride();
	}
	return img;
}/*}}}*/

template <class P>
Image<P> &&flip_vert(Image<P> &&img)/*{{{*/
{
	flip_vert_inplace(img);
	return std::move(img);
}/*}}}*/

namespace detail/*{{{*/
{
	template <int I, class P, class R>
	void save_component(const P &in, R &out)/*{{{*/
	{
	}/*}}}*/

	template <int I, class P, class R, class... VR>
	typename std::enable_if<I != P::dim>::type 
		save_component(const P &in, std::tuple<R, VR...> &out)/*{{{*/
	{
		using std::get;

		static_assert(I < color::traits::dim<P>::value, 
					  "Too much images to decompose into");

		typedef typename std::iterator_traits<R>::value_type out_type;

		*get<I>(out)++ 
			= color_cast<out_type>(get<I>(in));

		save_component<I+1>(in, out);
	}/*}}}*/

	template <int I, class P, class R>
	void load_component(P &in, R &out)/*{{{*/
	{
	}/*}}}*/

	template <int I, class P, class R, class... VR>
	typename std::enable_if<I != P::dim>::type 
		load_component(P &out, std::tuple<R, VR...> &in)/*{{{*/
	{
		using std::get;

		static_assert(I < color::traits::dim<P>::value, 
					  "Too much images to compose from");

		typedef typename color::traits::value_type<P>::type out_type;

		get<I>(out) 
			= color_cast<out_type>(*get<I>(in)++);

		load_component<I+1>(out, in);
	}/*}}}*/

	inline r2::usize get_size() // never called/*{{{*/
	{
		return r2::usize(0,0);
	}/*}}}*/

	template <class R, class... VR>
	r2::usize get_size(const R &head, const VR &...tail)/*{{{*/
	{
		r2::usize sz = head.size();

		if(sizeof...(VR) > 0)
		{
			if(sz != get_size(tail...))
				throw std::runtime_error("Images must have equal sizes");
		}

		return sz;
	}/*}}}*/

	inline void rescale_images(const r2::isize &sz) {}

	template <class I, class... VI>
	void rescale_images(const r2::isize &sz, I &head, VI &...tail)/*{{{*/
	{
		head = I(sz);
		rescale_images(sz, tail...);
	}/*}}}*/
}/*}}}*/

template <class P, class... R> 
void decompose(const Image<P> &img, Image<R> &...args)/*{{{*/
{
	static_assert(sizeof...(R) == color::traits::dim<P>::value, 
				  "Invalid number of arguments");

	auto out_size = detail::get_size(args...);
	if(img.size() != out_size)
	{
		if(out_size.is_zero())
			detail::rescale_images(img.size(), args...);
		else
			throw std::runtime_error("Images must have equal sizes");
	}

	std::tuple<typename Image<R>::iterator...> out(args.begin()...);

	for(auto in = img.cbegin(); in != img.cend(); ++in)
		detail::save_component<0>(*in, out);
}/*}}}*/

template <class P, class... R> 
Image<P> compose(const Image<R> &...args)/*{{{*/
{
	Image<P> img(detail::get_size(args...));

	std::tuple<typename Image<R>::const_iterator...> in(args.begin()...);

	for(auto out = img.begin(); out != img.end(); ++out)
		detail::load_component<0>(*out, in);

	return std::move(img);
}/*}}}*/

template <class P, class... R> 
Image<P> &compose(Image<P> &out, const Image<R> &...args)/*{{{*/
{
	return out = compose<P>(args...);
}/*}}}*/

template <class T>
Image<T> transpose(const Image<T> &srcimg)/*{{{*/
{
	Image<T> dstimg(srcimg.h(), srcimg.w());
	T *dst = dstimg.data();
	const T *src = srcimg.data();

	for(int y=srcimg.h(); y; --y)
	{
		T *dstcol = dst;
		for(int x=srcimg.w(); x; --x, dst+=dstimg.row_stride())
			*dst = *src++;
		dst = dstcol+1;
	}
	return std::move(dstimg);
}/*}}}*/

template <class T>
T maximum_difference(const Image<T> &srcimg)/*{{{*/
{
	std::pair<typename Image<T>::const_iterator,
		      typename Image<T>::const_iterator> 
		      	  diff = boost::minmax_element(srcimg.begin(), srcimg.end());
	return *diff.second - *diff.first;
}/*}}}*/

}} // namespace s3d::img

// $Id: image.hpp 3123 2010-09-09 19:02:52Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

