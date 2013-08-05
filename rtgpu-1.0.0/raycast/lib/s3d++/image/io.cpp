#include "pch.h"
#include "image.h"
#include "large_image.h"
#include "format/bitmap.h"
#include "format/jpeg.h"
#include "format/png.h"
#include "format/kro.h"
#include "io.h"

namespace s3d { namespace img {

namespace detail_img
{
std::unique_ptr<Image<>> load(std::istream &in, const parameters &params)
{
    if(is_format<BITMAP>(in))
    	return format_traits<BITMAP>::load(in, params);
	else if(is_format<KRO>(in))
    	return format_traits<KRO>::load(in, params);
#if HAS_PNG
    else if(is_format<PNG>(in))
    	return format_traits<PNG>::load(in, params);
#endif
#if HAS_JPEG
    else if(is_format<JPEG>(in))
    	return format_traits<JPEG>::load(in, params);
#endif
	else
		throw std::runtime_error("Format not supported");
}

template <class I>
void save_aux(const std::string &fname, I &&img, const parameters &params)
{
    unsigned dot = fname.rfind('.');
    if(dot == fname.npos)
		throw std::invalid_argument("Must specify file's extension");
    std::string ext = fname.substr(dot+1);

    if(handle_extension<BITMAP>(ext))
    	img::save<BITMAP>(fname, std::forward<I>(img), params);
	else if(handle_extension<KRO>(ext))
    	img::save<KRO>(fname, std::forward<I>(img), params);
#if HAS_PNG
    else if(handle_extension<PNG>(ext))
    	img::save<PNG>(fname, std::forward<I>(img), params);
#endif
#if HAS_JPEG
    else if(handle_extension<JPEG>(ext))
    	img::save<JPEG>(fname, std::forward<I>(img), params);
#endif
	else
		throw std::invalid_argument("Format not supported: "+ext);
}

void save(const std::string &fname, Image<> &&img, const parameters &params)
{
	save_aux(fname, std::move(img), params);
}

void save(const std::string &fname, const Image<> &img,const parameters &params)
{
	save_aux(fname, img, params);
}

auto load_large_impl(const std::string &fname, const parameters &params)
	-> std::unique_ptr<large_image_impl>
{
	unsigned dot = fname.rfind('.');
	if(dot == fname.npos)
		throw std::invalid_argument("Must specify file's extension");
	std::string ext = fname.substr(dot+1);

	if(handle_extension<BITMAP>(ext))
		return format_traits<BITMAP>::load_large(fname, params);
	else if(handle_extension<KRO>(ext))
		return format_traits<KRO>::load_large(fname, params);
#if HAS_PNG
	else if(handle_extension<JPEG>(ext))
		return format_traits<JPEG>::load_large(fname, params);
#endif
#if HAS_PNG
	else if(handle_extension<PNG>(ext))
		return format_traits<PNG>::load_large(fname, params);
#endif
	else
		throw std::invalid_argument("Format not supported: "+ext);
}

auto create_large_impl(const image_traits<> &traits, const r2::usize &s, 
				  const std::string &fname, const parameters &params)
	-> std::unique_ptr<large_image_impl>
{
	unsigned dot = fname.rfind('.');
	if(dot == fname.npos)
		throw std::invalid_argument("Must specify file's extension");
	std::string ext = fname.substr(dot+1);

	if(handle_extension<BITMAP>(ext))
		return format_traits<BITMAP>::create_large(traits, s, fname, params);
	else if(handle_extension<KRO>(ext))
		return format_traits<KRO>::create_large(traits, s, fname, params);
#if HAS_JPEG
	else if(handle_extension<JPEG>(ext))
		return format_traits<JPEG>::create_large(traits, s, fname, params);
#endif
#if HAS_PNG
	else if(handle_extension<PNG>(ext))
		return format_traits<PNG>::create_large(traits, s, fname, params);
#endif
	else
		throw std::invalid_argument("Format not supported: "+ext);
}

} // namespace detail_img

}}
