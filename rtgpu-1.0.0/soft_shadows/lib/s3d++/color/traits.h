#ifndef S3D_COLOR_TRAITS_H
#define S3D_COLOR_TRAITS_H

#include "fwd.h"
#include "../util/type_traits.h"

namespace s3d { namespace color
{

template <class T>
struct luminance_space;

namespace traits
{
	// has_alpha ------------------------------------
	template <class C> struct has_alpha : std::false_type {};
	template <class C, channel_position P> 
	struct has_alpha<color::alpha<C,P>> : std::true_type
	{
	};

	// invert_coords (RGB -> BGR)  ------------------------------------

	template <class C, class EN=void> 
	struct invert_coords : std::false_type
	{
	};

	template <class C> 
	struct invert_coords<C, typename std::enable_if<C::invert>::type> 
		: std::true_type
	{
	};

	// model ------------------------------------

	template <class C> 
	struct model
	{
		static const color::model value = std::is_arithmetic<C>::value 
											? color::model::GRAYSCALE
											: color::model::UNDEFINED;
	};

	// dim ------------------------------------

	template <class C, class EN=void> 
	struct dim;

	template <class C> 
	struct dim<C, typename std::enable_if<std::is_arithmetic<C>::value>::type>
	{
		static const size_t value = 1;
	};

	template <class C> 
	struct dim<C,typename std::enable_if<C::dim != 0>::type>
	{
		static const size_t value = C::dim;
	};

	// value_type ------------------------------------

	template <class C>
	struct value_type
	{
		typedef typename std::conditional
		<
			order<C>::value==0,
			std::identity<C>,
			s3d::value_type<C>
		>::type::type type;
	};

	// is_colorspace ------------------------------------

	template <class C>
	struct is_colorspace
	{
		static const bool value = model<C>::value != color::model::UNDEFINED;
	};

	// is_floating_point ------------------------------------

	template <class C>
	struct is_floating_point
	{
		static const bool value 
			= std::is_floating_point<typename value_type<C>::type>::value;
	};

	// is_integral ------------------------------------

	template <class C>
	struct is_integral
	{
		static const bool value 
			= std::is_integral<typename value_type<C>::type>::value;
	};

	// bpp ------------------------------------

	template <class C>
	struct bpp
	{
		static const size_t value = sizeof(C);
	};

	// depth ------------------------------------
	
	template <class C>
	struct depth
	{
		static const size_t value = bpp<C>::value*8 / dim<C>::value;
	};
	
	template <class C>
	struct is_homogeneous
	{
		// it's not exactly like this, but works in most cases
		static const bool value 
			= depth<C>::value*dim<C>::value == bpp<C>::value*8;
	};

	namespace detail
	{
		template <class C>
		struct space_impl
		{
			typedef typename C::space_type type;
		};
	}

	template <class C>
	struct space
	{
		typedef typename std::conditional
		<
			std::is_arithmetic<C>::value,
			std::identity<luminance_space<C>>,
			detail::space_impl<C>
		>::type::type type;
	};
} // namespace traits_def

using traits::is_colorspace;

template <>
struct color_traits<> // runtime-traits
{
private:
	template <class C, class EN=void>
	struct safe_dim
	{
		static const size_t value = 0;
	};
	template <class C>
	struct safe_dim<C,typename std::enable_if<traits::dim<C>::value>::type>
	{
		static const size_t value = traits::dim<C>::value;
	};

	template <class C, class EN=void>
	struct safe_depth
	{
		static const size_t value = 0;
	};
	template <class C>
	struct safe_depth<C,typename std::enable_if<traits::dim<C>::value>::type>
	{
		static const size_t value = traits::depth<C>::value;
	};

	template <class C, class EN=void>
	struct safe_is_homogeneous
	{
		static const bool value = false;
	};
	template <class C>
	struct safe_is_homogeneous<C,typename std::enable_if<traits::dim<C>::value>::type>
	{
		static const bool value = traits::is_homogeneous<C>::value;
	};
	

public:
	template <class C>
	color_traits(std::identity<C>)
		: value_type(typeid(typename traits::value_type<C>::type))
		, has_alpha(traits::has_alpha<C>::value)
		, model(traits::model<C>::value)
		, dim(safe_dim<C>::value)
		, depth(safe_depth<C>::value)
		, bpp(traits::bpp<C>::value)
		, invert(traits::invert_coords<C>::value)
		, is_floating_point(traits::is_floating_point<C>::value)
		, is_integral(traits::is_integral<C>::value)
		, is_homogeneous(safe_is_homogeneous<C>::value)
	{
	}
	virtual ~color_traits() {}

	const std::type_info &value_type;
	const bool has_alpha;
	const color::model model;
	const size_t dim;
	const size_t depth;
	const size_t bpp;
	const bool invert;
	const bool is_floating_point;
	const bool is_integral;
	const bool is_homogeneous;
};

template <class C>
struct color_traits : color_traits<>
{
	color_traits() : color_traits<>(std::identity<C>()) {}
};

} // namespace color

using color::color_traits;
using color::is_colorspace;

} // namespace s3d

#endif
