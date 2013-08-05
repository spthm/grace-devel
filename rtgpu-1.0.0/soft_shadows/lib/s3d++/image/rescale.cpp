#include "pch.h"
#include "rescale.h"
#include "image.h"
#include "cast.h"
#include "format/util.h" // pro image_cast_ptr

namespace p = std::placeholders;

namespace s3d { namespace img
{

auto rescale(const r2::isize &szdest, const Image<> &orig, 
			 const r2::ibox &bxorig)
	-> std::unique_ptr<Image<>>
{
	auto dest = orig.create(szdest);
	rescale_into(*dest, orig, bxorig);
	return std::move(dest);
};

auto rescale(const r2::isize &szdest, const Image<> &orig)
	-> std::unique_ptr<Image<>>
{
	auto dest = orig.create(szdest);
	rescale_into(*dest, orig);
	return std::move(dest);
}

namespace
{
	typedef mpl::vector<image, timage, rgb, rgba, rgb8, rgba8, rgb16, rgba16,
						bgr, bgra, bgr8, bgra8, bgr16, bgra16,
						luminance, ya, y8, y16, ya8, ya16> rescale_types;

	class dest_rescale_visitor : public visitor_impl<rescale_types>
	{
		class impl
		{
		public:
			template <class FROM>
			bool operator()(FROM &dest, const r2::box &bxdest,
							const Image<> &orig, const r2::ibox &bxorig)
			{
				rescale_into(dest, bxdest, *image_cast_ptr<FROM>(orig), bxorig);
				return true;
			}
		};
	public:
		dest_rescale_visitor(const r2::box &bxdest, 
						const Image<> &orig, const r2::ibox &bxorig) 
			: visitor_impl<rescale_types>(
				std::bind<bool>(impl(), p::_1, std::ref(bxdest),
										std::ref(orig), std::ref(bxorig)))
		{
		}
	};

	class orig_rescale_visitor : public const_visitor_impl<rescale_types>
	{
		class impl
		{
		public:
			template <class FROM>
			bool operator()(Image<> &dest, const r2::box &bxdest,
							const FROM &orig, const r2::ibox &bxorig)
			{
				auto pdest = image_cast_ptr<FROM>(dest);
				rescale_into(*pdest, bxdest, orig, bxorig);

				if(pdest.get() != &dest)
					dest = std::move(*dest.traits().image_cast(*pdest));
				return true;
			}
		};
	public:
		orig_rescale_visitor(Image<> &dest, const r2::box &bxdest, const r2::ibox &bxorig) 
			: const_visitor_impl<rescale_types>(
				std::bind<bool>(impl(), std::ref(dest), 
									std::ref(bxdest), p::_1, std::ref(bxorig)))
		{
		}
	};
}

void rescale_into(Image<> &dest, const r2::ibox &bxdest, 
				  const Image<> &orig, const r2::ibox &bxorig)
{
	if(dest.accept(dest_rescale_visitor(bxdest, orig, bxorig)))
		return;

	if(orig.accept(orig_rescale_visitor(dest, bxdest, bxorig)))
		return;

	auto resized_img = rescale(bxdest.size, image_cast<image>(orig), bxorig);

	copy_into(dest, bxdest, (const Image<> &)resized_img);
}


}} // namespace s3d::img
