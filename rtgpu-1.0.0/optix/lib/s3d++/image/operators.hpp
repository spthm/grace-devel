#include <algorithm>
#include "../util/gcc.h"

#if GCC_VERSION < 40500 || true
#	include <boost/bind.hpp>
#endif

namespace s3d { namespace img
{

namespace detail
{
	// In gcc-4.5 these structs don't need to be templates, std::result_of
	// is smart enough to deduce the return type

	template <class T, class U> 
	struct plus/*{{{*/
	{
		typedef decltype(std::declval<T>() + std::declval<U>()) result_type;

		result_type operator()(const T &x, const U &y) const
		{
			return x+y;
		}
	};/*}}}*/
	template <class T, class U> 
	struct minus/*{{{*/
	{
		typedef decltype(std::declval<T>() - std::declval<U>()) result_type;

		result_type operator()(const T &x, const U &y) const
		{
			return x-y;
		}
	};/*}}}*/
	template <class T, class U> 
	struct multiplies/*{{{*/
	{
//		typedef decltype(*((T *)NULL) * *((U *)NULL)) result_type;
		typedef decltype(std::declval<T>() * std::declval<U>()) result_type;

		result_type operator()(const T &x, const U &y) const
		{
			return x*y;
		}
	};/*}}}*/
	template <class T, class U> 
	struct divides/*{{{*/
	{
		//typedef decltype(*((T *)NULL) / *((U *)NULL)) result_type;
		typedef decltype(std::declval<T>() / std::declval<U>()) result_type;

		result_type operator()(const T &x, const U &y) const
		{
			return x/y;
		}
	};/*}}}*/
	template <class T> 
	struct negate/*{{{*/
	{
		//typedef decltype(- *((T *)NULL)) result_type;
		typedef decltype(-std::declval<T>()) result_type;

		auto operator()(const T &x) -> result_type
		{
			return -x;
		}
	};/*}}}*/

template <class I, class P, class EN=void>
struct image_pixel
{
	static const bool value = false;
};

template <class I, class P>
struct image_pixel<I,P,/*{{{*/
	typename std::enable_if<
		std::is_base_of<Image<>,I>::value && 
		std::is_convertible<P,typename I::value_type>::value
	>::type>
{
	static const bool value = true;
};/*}}}*/
template <class I, class P>
struct image_pixel<I,P,/*{{{*/
	typename std::enable_if<
		std::is_base_of<Image<>,I>::value && 
		!std::is_convertible<P,typename I::value_type>::value && 
		std::is_convertible<P,typename I::value_type::value_type>::value
	>::type>
{
	static const bool value = true;
};/*}}}*/
template <class I1, class I2>
struct image_image/*{{{*/
{
	static const bool value = 
		std::is_base_of<Image<>,I1>::value && std::is_base_of<Image<>,I2>::value;
};/*}}}*/
}

// I'm not using boost::operators in order to create rvalue-ref versions
// and avoid spurious temporary copying

// IMAGE x IMAGE operators

// member lvalue x const lvalue
template <class I1, class I2>
auto operator+=(I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&>::type 
{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");
	std::transform(img1.begin(), img1.end(), img2.begin(), img1.begin(),
				   detail::plus<typename I1::pixel_type, 
								typename I2::pixel_type>());
	return img1;
}/*}}}*/
template <class I1, class I2>
auto operator-=(I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&>::type
{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");
	std::transform(img1.begin(), img1.end(), img2.begin(), img1.begin(),
				   detail::minus<typename I1::pixel_type, 
								 typename I2::pixel_type>());
	return img1;
}/*}}}*/
template <class I1, class I2>
auto operator*=(I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&>::type 
{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");
	std::transform(img1.begin(), img1.end(), img2.begin(), img1.begin(),
				   detail::multiplies<typename I1::pixel_type, 
									  typename I2::pixel_type>());
	return img1;
}/*}}}*/
template <class I1, class I2>
auto operator/=(I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&>::type 

{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");
	std::transform(img1.begin(), img1.end(), img2.begin(), img1.begin(),
				   detail::divides<typename I1::pixel_type, 
								   typename I2::pixel_type>());
	return img1;
}/*}}}*/

// member rvalue x const lvalue
template <class I1, class I2>
auto operator+=(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type 

{
	return std::move(img1 += img2);
}/*}}}*/
template <class I1, class I2>
auto operator-=(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type

{
	return std::move(img1 -= img2);
}/*}}}*/
template <class I1, class I2>
auto operator/=(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 /= img2);
}/*}}}*/
template <class I1, class I2>
auto operator*=(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 *= img2);
}/*}}}*/

// rvalue x const lvalue
template <class I1, class I2>
auto operator+(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 += img2);
}/*}}}*/
template <class I1, class I2>
auto operator-(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 -= img2);
}/*}}}*/
template <class I1, class I2>
auto operator*(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 *= img2);
}/*}}}*/
template <class I1, class I2>
auto operator/(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 /= img2);
}/*}}}*/
template <class I>
auto operator-(I &&img)/*{{{*/
	-> typename std::enable_if<std::is_base_of<Image<>,I>::value, I&&>::type
{
	std::transform(img.begin(), img.end(), img.begin(),
				   detail::negate<typename I::pixel_type>());
	return std::move(img);
}/*}}}*/

template <class I1, class I2>
auto over(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type 

{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");

	auto it1 = img1.begin();
	auto it2 = img2.begin();
	for(; it1 != img1.end(); ++it1, ++it2)
		*it1 = over(*it1, *it2);

	return std::move(img1);
}/*}}}*/
template <class I1, class I2>
auto in(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type 

{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");

	auto it1 = img1.begin();
	auto it2 = img2.begin();
	for(; it1 != img1.end(); ++it1, ++it2)
		*it1 = in(*it1, *it2);

	return std::move(img1);
}/*}}}*/
template <class I1, class I2>
auto out(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type 

{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");

	auto it1 = img1.begin();
	auto it2 = img2.begin();
	for(; it1 != img1.end(); ++it1, ++it2)
		*it1 = out(*it1, *it2);

	return std::move(img1);
}/*}}}*/
template <class I1, class I2>
auto atop(I1 &&img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type 

{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");

	auto it1 = img1.begin();
	auto it2 = img2.begin();
	for(; it1 != img1.end(); ++it1, ++it2)
		*it1 = atop(*it1, *it2);

	return std::move(img1);
}/*}}}*/

// const lvalue x rvalue
template <class I1, class I2>
auto operator+(const I1 &img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I2&&>::type
{
	return std::move(img2 += img1);
}/*}}}*/
template <class I1, class I2>
auto operator-(const I1 &img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I2&&>::type
{
	return -std::move(img2) += img1;
}/*}}}*/
template <class I1, class I2>
auto operator*(const I1 &img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I2&&>::type
{
	return std::move(img2) *= img1;
}/*}}}*/
template <class I1, class I2>
auto operator/(const I1 &img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I2&&>::type
{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");
	std::transform(img1.begin(), img1.end(), img2.begin(), img2.begin(),
				   detail::divides<typename I1::pixel_type,
								   typename I2::pixel_type>());
	return std::move(img2);
}/*}}}*/

template <class I1, class I2>
auto over(const I1 &img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I2&&>::type 
{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");

	auto it1 = img1.begin();
	auto it2 = img2.begin();
	for(; it1 != img1.end(); ++it1, ++it2)
		*it2 = over(*it1, *it2);

	return std::move(img2);
}/*}}}*/
template <class I1, class I2>
auto in(const I1 &img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I2&&>::type 
{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");

	auto it1 = img1.begin();
	auto it2 = img2.begin();
	for(; it1 != img1.end(); ++it1, ++it2)
		*it2 = in(*it1, *it2);

	return std::move(img2);
}/*}}}*/
template <class I1, class I2>
auto out(const I1 &img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I2&&>::type 
{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");

	auto it1 = img1.begin();
	auto it2 = img2.begin();
	for(; it1 != img1.end(); ++it1, ++it2)
		*it2 = out(*it1, *it2);

	return std::move(img2);
}/*}}}*/
template <class I1, class I2>
auto atop(const I1 &img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I2&&>::type 
{
	if(img1.size() != img2.size())
		throw std::runtime_error("Image sizes mismatch");

	auto it1 = img1.begin();
	auto it2 = img2.begin();
	for(; it1 != img1.end(); ++it1, ++it2)
		*it2 = atop(*it1, *it2);

	return std::move(img2);
}/*}}}*/

// rvalue x rvalue (disambiguation)
template <class I1, class I2>
auto operator+(I1 &&img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 += img2);
}/*}}}*/
template <class I1, class I2>
auto operator-(I1 &&img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 -= img2);
}/*}}}*/
template <class I1, class I2>
auto operator*(I1 &&img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 *= img2);
}/*}}}*/
template <class I1, class I2>
auto operator/(I1 &&img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type
{
	return std::move(img1 /= img2);
}/*}}}*/

template <class I1, class I2>
auto over(I1 &&img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type 
{
	return over(std::move(img1), img2);
}/*}}}*/
template <class I1, class I2>
auto in(I1 &&img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type 
{
	return in(std::move(img1), img2);
}/*}}}*/
template <class I1, class I2>
auto out(I1 &&img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type 
{
	return out(std::move(img1), img2);
}/*}}}*/
template <class I1, class I2>
auto atop(I1 &&img1, I2 &&img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1&&>::type 
{
	return atop(std::move(img1), img2);
}/*}}}*/

// const lvalue x const lvalue
template <class I1, class I2>
auto operator+(const I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1>::type 
{
	return I1(img1) += img2;
}/*}}}*/
template <class I1, class I2>
auto operator-(const I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1>::type 
{
	return I1(img1) -= img2;
}/*}}}*/
template <class I1, class I2>
auto operator*(const I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1>::type 
{
	return I1(img1) *= img2;
}/*}}}*/
template <class I1, class I2>
auto operator/(const I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1>::type 
{
	return I1(img1) /= img2;
}/*}}}*/
template <class I>
auto operator-(const I &img)/*{{{*/
	-> typename std::enable_if<std::is_base_of<Image<>,I>::value, I>::type 
{
	I ret(img);

	std::transform(img.begin(), img.end(), ret.begin(), 
				   detail::negate<typename I::pixel_type>());
	return std::move(ret);
}/*}}}*/

template <class I1, class I2>
auto over(const I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1>::type 
{
	return over(I1(img1),img2);
}/*}}}*/
template <class I1, class I2>
auto in(const I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1>::type 
{
	return in(I1(img1),img2);
}/*}}}*/
template <class I1, class I2>
auto out(const I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1>::type 
{
	return out(I1(img1),img2);
}/*}}}*/
template <class I1, class I2>
auto atop(const I1 &img1, const I2 &img2)/*{{{*/
	-> typename std::enable_if<detail::image_image<I1,I2>::value, I1>::type 
{
	return atop(I1(img1),img2);
}/*}}}*/

// IMAGE x PIXEL operators

// member lvalue
template <class I, class P>
auto operator+=(I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&>::type
{

#if GCC_VERSION >= 40500 || true
	namespace p = std::placeholders;
	namespace a = std;
#else
	namespace a = boost;
	struct p
	{
		static const boost::arg<1> _1;
	};
#endif

	// Evita ambiguidade quando P != typename I::pixel_type mas é conversivel
	// p/ I::pixel_type (acontece com color::luminance, quando P = int, 
	// por exemplo
	typedef typename std::conditional
	<
		std::is_convertible<P,typename I::pixel_type>::value,
		typename I::pixel_type,
		P
	>::type type;
	
	std::transform(img.begin(), img.end(), img.begin(),
				a::bind(detail::plus<typename I::pixel_type,type>(), p::_1, c));
	return img;
}/*}}}*/
template <class I, class P>
auto operator-=(I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&>::type
{
#if GCC_VERSION >= 40500 || true
	namespace p = std::placeholders;
	namespace a = std;
#else
	namespace a = boost;
#endif
	// Evita ambiguidade quando P != typename I::pixel_type mas é conversivel
	// p/ I::pixel_type (acontece com color::luminance, quando P = int, 
	// por exemplo
	typedef typename std::conditional
	<
		std::is_convertible<P,typename I::pixel_type>::value,
		typename I::pixel_type,
		P
	>::type type;

	std::transform(img.begin(), img.end(), img.begin(),
			a::bind(detail::minus<typename I::pixel_type,P>(), p::_1, c));
	return img;
}/*}}}*/
template <class I, class P>
auto operator*=(I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&>::type
{
#if GCC_VERSION >= 40500 || true
	namespace p = std::placeholders;
	namespace a = std;
#else
	namespace a = boost;
#endif

	// Evita ambiguidade quando P != typename I::pixel_type mas é convertivel
	// p/ I::pixel_type (acontece com color::luminance, quando P = int, 
	// por exemplo
	typedef typename std::conditional
	<
		std::is_convertible<P,typename I::pixel_type>::value,
		typename I::pixel_type,
		P
	>::type type;

	std::transform(img.begin(), img.end(), img.begin(),
				   a::bind(detail::multiplies<typename I::pixel_type,type>(), p::_1, c));
	return img;
}/*}}}*/
template <class I, class P>
auto operator/=(I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&>::type
{
#if GCC_VERSION >= 40500 || true
	namespace p = std::placeholders;
	namespace a = std;
#else
	namespace a = boost;
#endif
	// Evita ambiguidade quando P != typename I::pixel_type mas é conversivel
	// p/ I::pixel_type (acontece com color::luminance, quando P = int, 
	// por exemplo
	typedef typename std::conditional
	<
		std::is_convertible<P,typename I::pixel_type>::value,
		typename I::pixel_type,
		P
	>::type type;

	std::transform(img.begin(), img.end(), img.begin(),
			a::bind(detail::divides<typename I::pixel_type,type>(), p::_1, c));
	return img;
}/*}}}*/

// member rvalue
template <class I, class P>
auto operator+=(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img += c);
}/*}}}*/
template <class I, class P>
auto operator-=(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img -= c);
}/*}}}*/
template <class I, class P>
auto operator*=(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img *= c);
}/*}}}*/
template <class I, class P>
auto operator/=(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img /= c);
}/*}}}*/

// rvalue
template <class I, class P>
auto operator+(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img += c);
}/*}}}*/
template <class I, class P>
auto operator-(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img -= c);
}/*}}}*/
template <class I, class P>
auto operator*(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img *= c);
}/*}}}*/
template <class I, class P>
auto operator/(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img /= c);
}/*}}}*/

template <class I, class P>
auto over(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type 
{
	for(auto it=img.begin(); it != img.end(); ++it)
		*it = over(*it, c);

	return std::move(img);
}/*}}}*/
template <class I, class P>
auto in(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type 
{
	for(auto it=img.begin(); it != img.end(); ++it)
		*it = in(*it, c);

	return std::move(img);
}/*}}}*/
template <class I, class P>
auto out(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type 
{
	for(auto it=img.begin(); it != img.end(); ++it)
		*it = out(*it, c);

	return std::move(img);
}/*}}}*/
template <class I, class P>
auto atop(I &&img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type 
{
	for(auto it=img.begin(); it != img.end(); ++it)
		*it = atop(*it, c);

	return std::move(img);
}/*}}}*/

// const lvalue
template <class I, class P>
auto operator+(const I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return I(img) += c;
}/*}}}*/
template <class I, class P>
auto operator-(const I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return I(img) -= c;
}/*}}}*/
template <class I, class P>
auto operator*(const I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return I(img) *= c;
}/*}}}*/
template <class I, class P>
auto operator/(const I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return I(img) /= c;
}/*}}}*/

template <class I, class P>
auto over(const I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return over(I(img), c);
}/*}}}*/
template <class I, class P>
auto in(const I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return in(I(img), c);
}/*}}}*/
template <class I, class P>
auto out(const I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return out(I(img), c);
}/*}}}*/
template <class I, class P>
auto atop(const I &img, const P &c)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return atop(I(img), c);
}/*}}}*/

// PIXEL x IMAGE operators

// rvalue
template <class I, class P>
auto operator+(const P &c, I &&img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img += c);
}/*}}}*/
template <class I, class P>
auto operator-(const P &c, I &&img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return -std::move(img) += c;
}/*}}}*/
template <class I, class P>
auto operator/(const P &c, I &&img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
#if GCC_VERSION >= 40500 || true
	namespace p = std::placeholders;
	namespace a = std;
#else
	namespace a = boost;
#endif

	// Evita ambiguidade quando P != typename I::pixel_type mas é conversivel
	// p/ I::pixel_type (acontece com color::luminance, quando P = int, 
	// por exemplo
	typedef typename std::conditional
	<
		std::is_convertible<P,typename I::pixel_type>::value,
		typename I::pixel_type,
		P
	>::type type;

	std::transform(img.begin(), img.end(), img.begin(),
			a::bind(detail::divides<type,typename I::pixel_type>(), c, p::_1));
	return std::move(img);
}/*}}}*/
template <class I, class P>
auto operator*(const P &c, I &&img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type
{
	return std::move(img *= c);
}/*}}}*/

template <class I, class P>
auto over(const P &c, I &&img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type 
{
	for(auto it=img.begin(); it != img.end(); ++it)
		*it = over(c, *it);

	return std::move(img);
}/*}}}*/
template <class I, class P>
auto in(const P &c, I &&img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type 
{
	for(auto it=img.begin(); it != img.end(); ++it)
		*it = in(c, *it);

	return std::move(img);
}/*}}}*/
template <class I, class P>
auto out(const P &c, I &&img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type 
{
	for(auto it=img.begin(); it != img.end(); ++it)
		*it = out(c, *it);

	return std::move(img);
}/*}}}*/
template <class I, class P>
auto atop(const P &c, I &&img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I&&>::type 
{
	for(auto it=img.begin(); it != img.end(); ++it)
		*it = atop(c, *it);

	return std::move(img);
}/*}}}*/

// const lvalue
template <class I, class P>
auto operator+(const P &c, const I &img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return img + c;
}/*}}}*/
template <class I, class P>
auto operator-(const P &c, const I &img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return -img + c;
}/*}}}*/
template <class I, class P>
auto operator/(const P &c, const I &img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	I out(img);

#if GCC_VERSION >= 40500 || true
	namespace p = std::placeholders;
	namespace a = std;
#else
	namespace a = boost;
#endif
	
	// Evita ambiguidade quando P != typename I::pixel_type mas é conversivel
	// p/ I::pixel_type (acontece com color::luminance, quando P = int, 
	// por exemplo
	typedef typename std::conditional
	<
		std::is_convertible<P,typename I::pixel_type>::value,
		typename I::pixel_type,
		P
	>::type type;

	std::transform(out.begin(), out.end(), out.begin(),
			a::bind(detail::divides<type,typename I::pixel_type>(), c, p::_1));
	return std::move(out);
}/*}}}*/
template <class I, class P>
auto operator*(const P &c, const I &img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return img * c;
}/*}}}*/

template <class I, class P>
auto over(const P &c, const I &img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return over(c, I(img));
}/*}}}*/
template <class I, class P>
auto in(const P &c, const I &img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return in(c, I(img));
}/*}}}*/
template <class I, class P>
auto out(const P &c, const I &img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return out(c, I(img));
}/*}}}*/
template <class I, class P>
auto atop(const P &c, const I &img)/*{{{*/
	-> typename std::enable_if<detail::image_pixel<I,P>::value, I>::type 
{
	return atop(c, I(img));
}/*}}}*/

}} // namespace s3d::img
