#include "pch.h"
#include "traits.h"
#include "large_image.h"
#include "image.h"

namespace s3d { namespace img
{

std::unique_ptr<Image<>> image_traits<>::create_image(const r2::usize &sz) const
{ 
	return std::unique_ptr<Image<>>(do_create_image(sz)); 
}

auto image_traits<>::load_large_image(std::unique_ptr<large_image_impl> impl) const
	-> std::unique_ptr<large_image<>> 
{ 
	return std::unique_ptr<large_image<>>(do_load_large_image(std::move(impl)));
}

std::unique_ptr<Image<>> image_traits<>::image_cast(const Image<> &img) const
{ 
	return std::unique_ptr<Image<>>(do_image_cast(img)); 
}

std::unique_ptr<Image<>> image_traits<>::image_cast(Image<> &&img) const
{ 
	return std::unique_ptr<Image<>>(do_image_cast(std::move(img))); 
}

}}
