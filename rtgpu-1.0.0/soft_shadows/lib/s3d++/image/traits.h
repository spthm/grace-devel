#ifndef S3D_IMAGE_TRAITS_H
#define S3D_IMAGE_TRAITS_H

#include "../color/traits.h"
#include "../color/cast.h"
#include "../math/r2/size.h"
#include "../util/pointer.h"
#include "large_image.h"
#include "cast.h"
#include "../mpl/bool.h"
#include "fwd.h"

namespace s3d { namespace img
{

namespace traits
{
	template <class I>
	struct value_type;
	
	template <class C>
	struct value_type<Image<C>>
	{
		typedef C type;
	};

	template <class I>
	struct pixel_type : value_type<I>
	{
	};

	template <class I>
	struct has_alpha : color::traits::has_alpha<typename value_type<I>::type>
	{
	};

	template <class I> 
	struct model : color::traits::model<typename value_type<I>::type>
	{
	};

	template <class I>
	struct dim : color::traits::dim<typename value_type<I>::type>
	{
	};

	template <class I>
	struct is_floating_point 
		: color::traits::is_floating_point<typename value_type<I>::type>
	{
	};

	template <class I>
	struct is_integral 
		: color::traits::is_integral<typename value_type<I>::type>
	{
	};

	template <class I>
	struct bpp : color::traits::bpp<typename value_type<I>::type>
	{
	};

	template <class I>
	struct depth : color::traits::depth<typename value_type<I>::type>
	{
	};

	template <class I>
	struct is_homogeneous 
		: color::traits::is_homogeneous<typename value_type<I>::type>
	{
	};

	template <class I>
	struct space : color::traits::space<typename value_type<I>::type>
	{
	};
}


class large_image_impl;

template <class I>
struct image_traits<I &> : image_traits<typename std::remove_cv<I>::type> {};

template <>
struct image_traits<> : color_traits<>
{
	typedef std::vector<uint8_t> typeless_color_type;

	template <class I>
	image_traits(std::identity<I>) 
		: color_traits<>(std::identity<typename traits::pixel_type<I>::type>()) {}

	std::unique_ptr<Image<>> create_image(const r2::usize &sz) const;

	// defined in traits.cpp, internal helper function
	auto load_large_image(std::unique_ptr<large_image_impl> impl) const
		-> std::unique_ptr<large_image<>>;

	template <class...PARAMS>
	auto create_large_image(const r2::usize &s, const std::string &fname,
							PARAMS &&...params) const
		-> std::unique_ptr<large_image<>>
	{
		return std::unique_ptr<large_image<>>(do_create_large_image(s,fname,
											{std::forward<PARAMS>(params)...}));
	}

	typeless_color_type color_cast(const color::radiance &c) const
		{ return do_color_cast(c); }

	std::unique_ptr<Image<>> image_cast(const Image<> &img) const;
	std::unique_ptr<Image<>> image_cast(Image<> &&img) const;

private:
	virtual Image<> *do_create_image(const r2::usize &sz) const = 0;
	virtual auto do_load_large_image(std::unique_ptr<large_image_impl> impl) const
		-> large_image<> * = 0;
	virtual auto do_create_large_image(const r2::usize &s, 
									   const std::string &fname,
									   const parameters &params) const
		-> large_image<> * = 0;
	virtual typeless_color_type do_color_cast(const color::radiance &c) const=0;
	virtual Image<> *do_image_cast(const Image<> &img) const = 0;
	virtual Image<> *do_image_cast(Image<> &&img) const = 0;
};

template <class I>
struct image_traits : image_traits<>
{
	static_assert(is_image<I>::value, "Must be an image!");

	image_traits() : image_traits<>(std::identity<I>()) {}

	std::unique_ptr<I> create_image(const r2::usize &sz) const
	{ 
		return std::static_pointer_cast<I>(image_traits<>::create_image(sz)); 
	}

	template <class FROM>
	auto color_cast(const FROM &c) const
		-> typename traits::pixel_type<I>::type 
	{
		return color_cast<typename traits::pixel_type<I>::type>(c);
	}

	template <class FROM>
	I image_cast(FROM &&img) const
	{ 
		return image_cast<I>(std::forward<FROM>(img));
	}

private:
	I *do_image_cast(const Image<> &img) const/*{{{*/
	{
		return img::image_cast<I>(img).move().release();
	}/*}}}*/
	I *do_image_cast(Image<> &&img) const/*{{{*/
	{
		return img::image_cast<I>(std::move(img)).move().release();
	}/*}}}*/
	
	auto do_load_large_image(std::unique_ptr<large_image_impl> impl) const/*{{{*/
		-> large_image<> *
	{ 
		return new large_image<I>(std::move(impl));
	}/*}}}*/
	auto do_create_large_image(const r2::usize &s, /*{{{*/
							   const std::string &fname,
							   const parameters &p) const
		-> large_image<> *
	{
		return create_large<I>(s, fname, p).move().release();
	}/*}}}*/

	virtual I *do_create_image(const r2::usize &sz) const
		{ return new I(sz); }

	template <bool B>
	static auto color_cast_impl(const color::radiance &_c, mpl::bool_<B>)
		-> typename std::enable_if<B==true,typeless_color_type>::type
	{
		auto c = color::color_cast<typename traits::pixel_type<I>::type>(_c);
		return typeless_color_type((uint8_t *)&c, (uint8_t *)&c+sizeof(c));
	}

	static auto color_cast_impl(const color::radiance &_c, mpl::bool_<false>)
		-> typeless_color_type
	{
		throw std::runtime_error("Cannot cast to a type that's not a colorspace");
	}

	virtual auto do_color_cast(const color::radiance &_c) const
		-> typeless_color_type 
	{
		return color_cast_impl(_c, 
		    mpl::bool_<is_colorspace<typename traits::pixel_type<I>::type>::value>());
	}
};

}} // namespace s3d::img

#endif
