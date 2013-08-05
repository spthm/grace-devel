#ifndef S3D_IMAGE_CONVOLUTION_H
#define S3D_IMAGE_CONVOLUTION_H

#include "fwd.h"
#include "kernel.h"

namespace s3d { namespace img {

luminance convolve(const luminance &src, const kernel &K);
luminance convolve(const luminance &src, const r2::ibox &area, 
				   const kernel &K);

packed_image convolve(const packed_image &src, const kernel &K);

}} // namespace s3d::img

#endif
