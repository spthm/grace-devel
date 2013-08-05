#ifndef S3D_PACKED_IMAGE_H
#define S3D_PACKED_IMAGE_H

#include "cast.h"
#include "../color/rgb.h" // for v4f handling
#include "image.h"

namespace s3d { namespace img {

class packed_image : public Image<v4f>
{
	typedef Image<v4f> base;
public:
	packed_image(packed_image &&that) 
		: base(std::move(that)), m_height(that.m_height) {}
	packed_image(const packed_image &that)
		: base(that), m_height(that.m_height) {}

	packed_image(const luminance &img);
	packed_image();
	packed_image(const r2::usize &s);
	packed_image(size_t w, size_t h);
	operator luminance() const;

	packed_image &operator=(packed_image &&that);
	packed_image &operator=(const packed_image &that);

	r2::usize size() const { return {w(), h()}; }

	size_t h() const { return m_height; }
	size_t packed_height() const { return base::h(); }

	DEFINE_MOVABLE(packed_image);
	DEFINE_CLONABLE(packed_image);
	DEFINE_CREATABLE(packed_image);
	DEFINE_CREATABLE(packed_image, const r2::usize &);
private:
	virtual img::image do_conv_to_radiance() const;
	size_t m_height;
};

template <>
struct is_image<packed_image>
{
	static const bool value = true;
};

template <class FROM>
struct cast_def<FROM, packed_image>
{
	static packed_image map(const FROM &f)
	{
		return image_cast<luminance>(f);
	};
};

template <class TO>
struct cast_def<packed_image, TO>
{
	static TO map(const packed_image &f)
	{
		return image_cast<TO>(static_cast<luminance>(f));
	};
};

packed_image double_scale(const packed_image &img);
packed_image half_scale(const packed_image &img);
packed_image transpose(const packed_image &img);
float maximum_difference(const packed_image &img);

packed_image upsample(const packed_image &img);
packed_image downsample(const packed_image &img);

#if 0

void copy_into(packed_image &dest, const packed_image &orig);

void copy_into(packed_image &dest, const r2::ibox &bxdest, 
			   const packed_image &orig);

void copy_into(packed_image &dest, const packed_image &orig, 
			   const r2::ibox &bxorig);

void copy_into(packed_image &dest, const r2::ibox &bxdest, 
	           const packed_image &orig, const r2::ibox &bxorig);

packed_image copy(const packed_image &orig, const r2::ibox &bxorig);
#endif

}}


#endif
