#ifndef S3D_IMAGE_CAST_DEF_H
#define S3D_IMAGE_CAST_DEF_H

#include "../color/cast.h"
#include "../color/color.h" // needed for dynamic image_cast
#include "../util/visitor.h"
#include "../mpl/vector_fwd.h"

namespace s3d { namespace img
{

template <class FROM, class TO>
struct cast_def
{
	static_assert(!std::is_same<FROM,Image<>>::value, 
		"You must include dynamic_cast.h for conversions from Image<>");

	static const bool is_convertible = 
		is_explicitly_convertible<FROM, TO>::value &&
		!std::is_same<TO,image>::value && !std::is_same<TO,timage>::value;

	static const bool valid 
		= color::can_cast<typename traits::value_type<FROM>::type, 
						  typename traits::value_type<TO>::type>::value;

	// FROM can be converted to TO implicitly
	template <class DUMMY=int>
	static auto map(const FROM &c)
		-> typename std::enable_if<sizeof(DUMMY) && valid && 
								   is_convertible,TO>::type
	{
		return static_cast<TO>(c);
	}

	template <class DUMMY=int>
	static auto map(const FROM &c)
		-> typename std::enable_if<sizeof(DUMMY) && valid && 
								   !is_convertible,TO>::type
	{
		TO out(c.size());

		typename TO::value_type *d = out.data(),
				                *end = d+out.length();
		const typename FROM::value_type *o = c.data();

		while(d != end)
			*d++ = color_cast<typename TO::value_type>(*o++);

		return out;
	}

	template <class DUMMY=int>
	static auto map(FROM &&c)
		-> typename std::enable_if<sizeof(DUMMY) && valid, TO>::type
	{
		// when gcc support delegating ctors, 
		// this ugliness will be gone for good
		TO to;
		to.move_ctor(std::move(c));
		return to;
	}

	template <class DUMMY=int>
	static auto map(const FROM &c)
		-> typename std::enable_if<sizeof(DUMMY) && !valid, TO>::type
	{
		throw std::runtime_error("Invalid tone mapping");
	}

	template <class DUMMY=int>
	static auto map(FROM &&c)
		-> typename std::enable_if<sizeof(DUMMY) && !valid, TO>::type
	{
		throw std::runtime_error("Invalid tone mapping");
	}
};

template <class FROM>
struct cast_def<FROM, Image<>>
{
	static const Image<> &map(const FROM &img)
	{
		return img;
	}
};

namespace detail
{
	typedef mpl::vector<image, timage, rgb, rgba, rgb8, rgba8, rgb16, rgba16,
						bgr, bgra, bgr8, bgra8, bgr16, bgra16,
						luminance, ya, y8, y16, ya8, ya16> conv_types;

	template <class TO>
	class conv_visitor
		: public visitor_impl<conv_types>
		, public const_visitor_impl<conv_types>
	{
		class impl
		{
		public:
			impl(TO &dest) : m_dest(dest) {}

			template <class FROM>
			bool operator()(FROM &&visitable)
			{
				m_dest = image_cast<TO>(std::forward<FROM>(visitable));
				return true;
			}
		private:
			TO &m_dest;
		};
	public:
		conv_visitor(TO &dest) 
			: visitor_impl<conv_types>(impl(dest))
			, const_visitor_impl<conv_types>(impl(dest)) {}
	};
}


template <class CTO>
struct cast_def<Image<>, Image<CTO>>
{
	typedef Image<CTO> TO;

	static TO map(const Image<> &img)
	{
		if(auto *_img = dynamic_cast<const TO *>(&img))
			return *_img;
		else
		{
			TO dest;
			if(!img.accept(detail::conv_visitor<TO>(dest)))
			{
				if(traits::has_alpha<TO>::value)
					dest = image_cast<TO>(img.to_radiance_alpha());
				else
					dest = image_cast<TO>(img.to_radiance());
			}

			assert(img.size() == dest.size());
			return dest;
		}
	}
	static TO map(Image<> &&img)
	{
		if(auto *_img = dynamic_cast<TO *>(&img))
			return std::move(*_img);
		else
		{
			auto oldsize = img.size();
			TO dest;
			if(!img.move_accept(detail::conv_visitor<TO>(dest)))
			{
				if(traits::has_alpha<TO>::value)
					dest = image_cast<TO>(img.move_to_radiance_alpha());
				else
					dest = image_cast<TO>(img.move_to_radiance());
			}

			assert(oldsize == dest.size());
			return dest;
		}
	}
};

}} 

#endif
