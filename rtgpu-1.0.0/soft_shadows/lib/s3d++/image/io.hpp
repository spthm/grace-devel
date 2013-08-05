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
#include <cerrno>
#include <functional>
#include <fstream>
#include <boost/iostreams/stream.hpp>
#include "params.h"
#include "../util/gcc.h"
#include "cast.h"

namespace s3d { namespace img
{

namespace detail_img
{

std::unique_ptr<Image<>> load(std::istream &in, const parameters &params);
void save(const std::string &fname, const Image<> &img, const parameters &);
void save(const std::string &fname, Image<> &&img, const parameters &);

template <class I>
void update_parameters(parameters &pars, std::identity<Image<I>>)/*{{{*/
{
	static_assert(sizeof(I), "Oops, you must pass pixel types, not images");
}/*}}}*/

template <class C, color::channel_position P>
void update_parameters(parameters &pars, std::identity<color::alpha<C,P>>)/*{{{*/
{
	pars.insert(alpha = true);
	update_parameters(pars,std::identity<Image<C>>());
}/*}}}*/

template <class T>
inline void update_parameters(parameters &pars, std::identity<T>)/*{{{*/
{
	if(color::traits::model<T>::value == model::GRAYSCALE)
	{
		pars.insert(grayscale = true);
		pars.insert(bpp = (std::is_integral<T>::value ? sizeof(T) : 16));
	}
}/*}}}*/

template <template<class> class C, class T>
inline void update_parameters(parameters &pars, std::identity<C<T>>)/*{{{*/
{
	pars.insert(bpp = (std::is_integral<T>::value ? sizeof(T) : 16));
}/*}}}*/

}

template <image_format F> bool is_format(const uint8_t *data, size_t len)/*{{{*/
{
    namespace io = boost::iostreams;

    io::stream<io::array> in((const char *)data, len);
    return format_traits<F>::is_format(in);
}/*}}}*/
template <image_format F> bool is_format(const std::string &fname)/*{{{*/
{
	std::ifstream in(fname.c_str(), std::ios::binary);
	if(!in)
		throw std::runtime_error(fname + ": " + strerror(errno));
    return format_traits<F>::is_format(in);
}/*}}}*/
template <image_format F> bool is_format(std::istream &in)/*{{{*/
{
    return format_traits<F>::is_format(in);
}/*}}}*/

template <image_format F> bool handle_extension(const std::string &ext)/*{{{*/
{
	return format_traits<F>::handle_extension(ext);
}/*}}}*/

template <class I, class...PARAMS> 
I load(std::istream &in, PARAMS &&...params)/*{{{*/
{
	parameters p{std::forward<PARAMS>(params)...};
	detail_img::update_parameters(p, std::identity<typename I::pixel_type>());

	return image_cast<I>(std::move(*detail_img::load(in, p)));
}/*}}}*/

template <class I, class...PARAMS> 
I load(const std::string &fname, PARAMS &&...params)/*{{{*/
{
	std::ifstream in(fname.c_str(), std::ios::binary);
	if(!in)
		throw std::runtime_error(fname + ": " + strerror(errno));

	return load<I>(in, std::forward<PARAMS>(params)...);
}/*}}}*/

template <class I, class...PARAMS> 
I load(const uint8_t *data, size_t len, PARAMS &&...params)/*{{{*/
{
    namespace io = boost::iostreams;

    io::stream<io::array> in((char *)data, len);
	return load<I>(in, std::forward<PARAMS>(params)...);
}/*}}}*/

template <class...PARAMS> 
std::unique_ptr<Image<>> load(const std::string &fname, PARAMS &&...params)/*{{{*/
{
	std::ifstream in(fname.c_str(), std::ios::binary);
	if(!in)
		throw std::runtime_error(fname + ": " + strerror(errno));

	return load(in, std::forward<PARAMS>(params)...);
}/*}}}*/

template <class...PARAMS> 
std::unique_ptr<Image<>> load(const uint8_t *data, size_t len, PARAMS &&...params)/*{{{*/
{
    namespace io = boost::iostreams;

    io::stream<io::array> in((char *)data, len);
	return load(in, std::forward<PARAMS>(params)...);
}/*}}}*/

template <class...PARAMS>
std::unique_ptr<Image<>> load(std::istream &in, PARAMS &&...params)/*{{{*/
{
	return detail_img::load(in, {std::forward<PARAMS>(params)...});
}/*}}}*/

template <image_format F, class I, class...PARAMS> 
void save(std::ostream &out, I &&img, PARAMS &&...params)/*{{{*/
{
	format_traits<F>::save(out, std::forward<I>(img), 
						   parameters{std::forward<PARAMS>(params)...});
}/*}}}*/
template <image_format F, class I, class...PARAMS> 
void save(const std::string &fname, I &&img, PARAMS &&...params)/*{{{*/
{
	std::ofstream out(fname.c_str(), std::ios::binary);
	if(!out)
		throw std::runtime_error(fname + ": " + strerror(errno));

	try
	{
		save<F,I>(out, std::forward<I>(img), std::forward<PARAMS>(params)...);
	}
	catch(...)
	{
		::unlink(fname.c_str());
		throw;
	}
}/*}}}*/

template <class...PARAMS> 
void save(const std::string &fname, const Image<> &img, PARAMS &&...params)/*{{{*/
{
	detail_img::save(fname, img, {std::forward<PARAMS>(params)...});
}/*}}}*/

template <class I, class...PARAMS> 
void save(const std::string &fname, I &&img, PARAMS &&...params)/*{{{*/
{
	static_assert(is_image<I>::value, "Parameter must be an image");

	detail_img::save(fname, std::forward<I>(img), 
					 {std::forward<PARAMS>(params)...});
}/*}}}*/

}} // s3d::img

// $Id: io.hpp 2978 2010-08-19 02:03:55Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

