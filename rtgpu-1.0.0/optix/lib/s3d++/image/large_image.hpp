#include "io.h"
#include "cast.h"

namespace s3d { namespace img
{

namespace detail_img
{
	auto create_large_impl(const image_traits<> &traits, const r2::usize &s, 
					  const std::string &fname, const parameters &params)
		-> std::unique_ptr<large_image_impl>;

	auto load_large_impl(const std::string &fname, const parameters &params)
		-> std::unique_ptr<large_image_impl>;
}

template <class I, class...PARAMS>
large_image<I> load_large(const std::string &fname, PARAMS &&...params)
{
	auto impl = detail_img::load_large_impl(fname, 
											{std::forward<PARAMS>(params)...});
	return large_image<I>(std::move(impl));
}

template <class I, class...PARAMS>
large_image<I> create_large(const r2::usize &s, const std::string &fname,
							PARAMS &&...params)
{
	auto impl = detail_img::create_large_impl(I::traits(), s, fname,
											{std::forward<PARAMS>(params)...});

	return large_image<I>(std::move(impl));
}

template <class...PARAMS>
std::unique_ptr<large_image<>> load_large(const std::string &fname,
										  PARAMS &&...params)
{
	auto impl = detail_img::load_large_impl(fname,
											{std::forward<PARAMS>(params)...});

	auto &traits = impl->traits();
	return traits.load_large_image(std::move(impl));
}

template <class I>
I large_image<I>::read(const r2::ibox &bounds) const
{
	std::unique_ptr<Image<>> img = base::read(bounds);
	if(auto *i = dynamic_cast<I *>(img.get()))
		return std::move(*i);
	else
		return image_cast<I>(std::move(*img));
}
}} // namespaec s3d::img
