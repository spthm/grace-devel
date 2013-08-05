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

// $Id: image.h 3123 2010-09-09 19:02:52Z rodolfo $
// vim: nocp:ci:sts=4:fdm=marker:fmr={{{,}}}
// vi: ai:sw=4:ts=8

#ifndef S3D_IMAGE_IMAGE_H
#define S3D_IMAGE_IMAGE_H

#include "config.h"
#include <iosfwd>
#include <memory>
#include <climits>
#include <vector>
#include <array>
#include "../util/type_traits.h"
#include "../math/r2/size.h"
#include "../math/r2/box.h"
#include "../util/unique_ptr.h"
#include "../util/clonable.h"
#include "../util/creatable.h"
#include "../util/movable.h"
#include "../util/visitor.h"
#include "../color/alpha.h"
#include "fwd.h"
#include "traits.h"

namespace s3d { namespace img
{

template <>
class Image<> : public clonable, public movable, public visitable/*{{{*/
		      , public creatable<>, public creatable<const r2::usize &>
{
    Image &operator=(const Image &that);
protected:
    Image(void *data, size_t capacity, const math::r2::usize &sz, 
    	  const image_traits<> &traits, size_t row_stride = 0);
    Image(Image &&that);
    Image(const Image &that);

	template <class> friend class Image;

    Image &move_assign_impl(Image &&that);
    Image &copy_assign_impl(const Image &that);

public:
    virtual ~Image() {}

    uint8_t *data() 
		{ return reinterpret_cast<uint8_t *>(m_data); }
    const uint8_t *data() const 
		{ return reinterpret_cast<const uint8_t *>(m_data); }

    size_t length() const { return h()*row_stride(); }
    size_t capacity() const { return m_capacity; }

    void clear() { std::fill(data(), data()+length(), 0); }

    size_t row_stride() const { return m_row_stride; }
    size_t col_stride() const { return traits().bpp; }
    size_t bpp() const { return traits().bpp; } // bytes por pixel

    size_t byte_row_stride() const { return row_stride(); }
    size_t byte_col_stride() const { return col_stride(); }

    uint8_t *pixel_ptr_at(const r2::ipoint &pos);
    const uint8_t *pixel_ptr_at(const r2::ipoint &pos) const;

    template <class P> bool has_pixel_type() const;

    const image_traits<> &traits() const { assert(m_traits); return *m_traits; }

    const math::r2::usize &size() const { return m_size; }
    size_t w() const { return m_size.w; }
    size_t h() const { return m_size.h; }

	math::r2::ubox region() const { return math::r2::ubox(0,0,w(),h()); }

    Image &operator=(Image &&that);

    img::image move_to_radiance();
    img::timage move_to_radiance_alpha();

    img::image to_radiance() const;
    img::timage to_radiance_alpha() const;

    friend size_t memory_used(const Image &im)
		{ return im.length()+sizeof(im); }

    DEFINE_PURE_CLONABLE(Image);
    DEFINE_PURE_MOVABLE(Image);
    DEFINE_PURE_CREATABLE(Image);
    DEFINE_PURE_CREATABLE(Image, const r2::usize &);
private:
    virtual img::image do_move_to_radiance() = 0;
    virtual img::timage do_move_to_radiance_alpha() = 0;
    virtual img::image do_conv_to_radiance() const = 0;
    virtual img::timage do_conv_to_radiance_alpha() const = 0;

	void *m_data;
	size_t m_capacity; // in bytes, might be greater than space used by image
    r2::usize m_size;
    size_t m_row_stride;
    const image_traits<> *m_traits;
};/*}}}*/

template <class P> class Image 
	: public Image<>
{
protected:
	template <class IMG>
	struct row_proxy/*{{{*/
	{
	public:
		row_proxy(IMG &img, int row)
			: m_image(img), m_row(row) {}

		typename IMG::pixel_type &operator[](int col) const
		{
			return m_image.pixel_at({col, m_row});
		}
	private:
		IMG &m_image;
		int m_row;
	};/*}}}*/
	template <class IMG>
	struct const_row_proxy/*{{{*/
	{
	public:
		const_row_proxy(const IMG &img, int row)
			: m_image(img), m_row(row) {}

		const typename IMG::pixel_type &operator[](int col) const
		{
			return m_image.pixel_at({col, m_row});
		}
	private:
		const IMG &m_image;
		int m_row;
	};/*}}}*/

public:
    typedef P value_type;
    typedef P pixel_type;
    typedef P *iterator;
    typedef const P *const_iterator;

	template <class T>
	void move_ctor(Image<T> &&img, 
		  typename std::enable_if<!std::is_same<void,P>::value && 
						          sizeof(P) <= sizeof(T)>::type* =NULL);

	template <class T>
	void move_ctor(Image<T> &&img, 
		  typename std::enable_if<!std::is_same<void,P>::value && 
						           (sizeof(P) > sizeof(T))>::type* =NULL);

	template <class> friend class Image;
	//template <class, class> friend class cast_def;

	Image();
    Image(Image &&that);
    Image(const Image &that);

    Image(std::unique_ptr<P[]> data, const r2::usize &sz, size_t row_stride=0);
    Image(const r2::usize &sz, size_t row_stride=0);
    // needed to avoid ambiguity when using initializer list for size
    Image(size_t w, size_t h, size_t row_stride=0);

	// Conversion from/to any image to/from radiance/radiance_alpha is implicit
	template <class T>
    Image(const Image<T> &that, 
    	  typename std::enable_if
		  <	  
			!std::is_same<P,T>::value && std::is_convertible<T,P>::value &&
			(std::is_same<T,color::radiance>::value ||
			 std::is_same<T,color::radiance_alpha>::value ||
			 std::is_same<P,color::radiance>::value ||
			 std::is_same<P,color::radiance_alpha>::value)
	      >::type* =NULL);


	template <class T>
    Image(Image<T> &&that, 
    	  typename std::enable_if
		  <	  
			!std::is_same<P,T>::value && std::is_convertible<T,P>::value &&
			(std::is_same<T,color::radiance>::value ||
			 std::is_same<T,color::radiance_alpha>::value ||
			 std::is_same<P,color::radiance>::value ||
			 std::is_same<P,color::radiance_alpha>::value)
	      >::type* =NULL);

    ~Image() { delete[] data(); }

    Image &operator=(const Image &that);
    Image &operator=(Image &&that);

    P *data() { return reinterpret_cast<P *>(m_data); }
    const P *data() const { return reinterpret_cast<const P *>(m_data); }

    static const image_traits<Image> &traits() 
	{ 
		static const image_traits<Image> traits;
		return traits;
	}

    size_t length() const { return h()*row_stride(); }
    size_t row_stride() const { return m_row_stride; }
    size_t col_stride() const { return 1; }

    bool contains(const r2::ipoint &p) const
		{ return p.x>=0 && p.x<(int)w() && p.y>=0 && p.y<(int)h(); }

    row_proxy<Image> operator[](int row) 
		{ return row_proxy<Image>(*this,row); }
    const_row_proxy<Image> operator[](int row) const 
		{ return const_row_proxy<Image>(*this, row); }

    P &pixel_at(const r2::ipoint &pos);
    const P &pixel_at(const r2::ipoint &pos) const;

    P *pixel_ptr_at(const r2::ipoint &pos);
    const P *pixel_ptr_at(const r2::ipoint &pos) const;

	using Image<>::clear;
	template <class C>
    void clear(const C &color);

    iterator begin() { return data(); }
    iterator end() { return data()+length(); }
    const_iterator begin() const { return data(); }
    const_iterator end() const { return data()+length(); }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const  { return end(); }


    DEFINE_CLONABLE(Image);
    DEFINE_MOVABLE(Image);
    DEFINE_CREATABLE(Image);
    DEFINE_CREATABLE(Image, const r2::usize &);
private:
	size_t m_row_stride;

	image do_conv_to_radiance() const /*{{{*/
	{ 
		return image_cast<image>(*this);
	}/*}}}*/
	timage do_conv_to_radiance_alpha() const /*{{{*/
	{ 
		return image_cast<timage>(*this);
	}/*}}}*/

	image do_move_to_radiance() /*{{{*/
	{ 
		return image_cast<image>(std::move(*this));
	}/*}}}*/
	timage do_move_to_radiance_alpha() /*{{{*/
	{ 
		return image_cast<timage>(std::move(*this));
	}/*}}}*/

    virtual bool do_move_accept(visitor_base &&visitor);
    virtual bool do_accept(visitor_base &&visitor);
    virtual bool do_accept(visitor_base &&visitor) const;
};

template <class T>
struct is_image
{
	static const bool value = false;
};

template <class T>
struct is_image<Image<T>>
{
	static const bool value = true;
};

template <class T>
struct is_image<T &> : is_image<T> {};

template <class P>
r2::ibox bounds(const Image<P> &img);

template <class P>
r2::point centroid(const Image<P> &img);

template <class P>
P bicubic_pixel(const Image<P> &img, const r2::point &pos);

template <class P>
P bilinear_pixel(const Image<P> &img, const r2::point &pos);

template <class P>
auto gradient_pixel(const Image<P> &img, const r2::ipoint &pos)
	-> Vector<typename difference_type<P>::type,2>;

template <class P> real aspect(const Image<P> &img);

template <class P>
Image<P> copy(const Image<P> &orig, const r2::ibox &bxorig);

#if 0
template <class P>
Image<P> &flip_horiz_inplace(Image<P> &img);
#endif

template <class P>
Image<P> &flip_vert_inplace(Image<P> &img);

template <class P>
Image<P> &&flip_vert(Image<P> &&img);

auto rescale(const r2::isize &szdest, const Image<> &orig, 
			 const r2::ibox &bxorig)
	-> std::unique_ptr<Image<>>;

auto rescale(const r2::isize &szdest, const Image<> &orig)
	-> std::unique_ptr<Image<>>;

auto copy(const Image<> &orig, const r2::ibox &bxorig)
	-> std::unique_ptr<Image<>>;

template <class P>
void copy_into(Image<P> &dest, const Image<P> &orig);

template <class P>
void copy_into(Image<P> &dest, const r2::ibox &bxdest, const Image<P> &orig);

template <class P>
void copy_into(Image<P> &dest, const Image<P> &orig, const r2::ibox &bxorig);

template <class P>
void copy_into(Image<P> &dest, const r2::ibox &bxdest, 
			   const Image<P> &orig, const r2::ibox &bxorig);

// defined in image.cpp
void copy_into(Image<> &dest, const r2::ibox &bxdest, 
			   const Image<> &orig, const r2::ibox &bxorig);

image apply_gamma(const image &orig, real g);
void apply_gamma_into(image &orig, real g);

template <class P, class... R> 
void decompose(const Image<P> &img, Image<R> &...args);

template <class P, class... R> 
Image<P> compose(const Image<R> &...args);

template <class P, class... R> 
Image<P> &compose(Image<P> &out, const Image<R> &...args);

image compose(const luminance &x, const luminance &y, const luminance &z);

typedef std::vector<img::luminance> grayscale_pyramid;
typedef std::array<std::vector<img::luminance>,3> color_pyramid;

color_pyramid laplacian_pyramid(const image &img, int levels=INT_MAX);
image laplacian_reconstruct(const color_pyramid &pyr);

grayscale_pyramid laplacian_pyramid(const luminance &img, int levels=INT_MAX);
grayscale_pyramid laplacian_pyramid(luminance &&img, int levels=INT_MAX);
luminance laplacian_reconstruct(const grayscale_pyramid &pyr);

image gaussian_blur(const image &img, float sigma, int dim=0);

}} // namespace s3d::img

#include "image.hpp"
#include "operators.hpp"
#include "cast_def.hpp"

#endif

// $Id: image.h 3123 2010-09-09 19:02:52Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

