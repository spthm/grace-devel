#ifndef S3D_IMAGE_CAST_H
#define S3D_IMAGE_CAST_H

#include "../util/concepts.h"
#include "../util/type_traits.h"
#include "fwd.h"

namespace s3d { namespace img
{

template <class FROM, class TO> struct cast_def;

// indirection needed to avoid evaluation of cast_def without image types
namespace detail_img
{
	template<class FROM, class TO>
	struct image_cast_ret
	{
		static_assert(!std::is_reference<TO>::value, 
					  "TO must not be a reference");

	   typedef decltype(cast_def<typename remove_cv_ref<FROM>::type, 
								     TO>::map(std::declval<FROM>())) type;
	};
}

template <class TO, class FROM>
auto image_cast(FROM &&c)
	-> typename requires<is_image<typename remove_cv_ref<FROM>::type>,
	   typename detail_img::image_cast_ret<FROM,TO>>::type::type
{
	static_assert(is_image<TO>::value, "TO must be an image");

	return cast_def<typename remove_cv_ref<FROM>::type, 
					    TO>::map(std::forward<FROM>(c));
}

}} // namespace s3d::img

#endif
