#ifndef S3D_IMAGE_FORMAT_UTIL_H
#define S3D_IMAGE_FORMAT_UTIL_H

#include "../cast.h"
#include "../../util/gcc.h"
#include "../../util/pointer.h"

namespace s3d { namespace img
{

template <class TO, class FROM>
std::shared_ptr<const TO> image_cast_ptr(const FROM &img)
{
	if(auto *pimg = dynamic_cast<const TO *>(&img))
		return {pimg, null_deleter};
	else
#if GCC_VERSION >= 40500
		return image_cast<TO>(img).move();
#else
		return std::shared_ptr<const TO>(image_cast<TO>(img).move().release());
#endif
}

template <class TO, class FROM>
std::shared_ptr<TO> image_cast_ptr(FROM &&img)
{
#if GCC_VERSION >= 40500
	if(auto *pimg = dynamic_cast<TO *>(&img))
		return pimg->move();
	else
		return image_cast<TO>(std::move(img)).move();
#else
	if(auto *pimg = dynamic_cast<TO *>(&img))
		return std::shared_ptr<TO>(pimg->move().release());
	else
		return std::shared_ptr<TO>(image_cast<TO>(std::move(img)).move().release());
#endif
}

}}

#endif
