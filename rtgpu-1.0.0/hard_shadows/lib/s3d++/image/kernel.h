#ifndef S3D_IMAGE_KERNEL_H
#define S3D_IMAGE_KERNEL_H

#include "../math/real.h"

namespace s3d { namespace img {

typedef std::vector<real> kernel;

kernel gaussian_kernel(float sigma, int dim=0);
kernel blur_kernel(real a=0.4);

template <class T> inline float gaussian_weight(T x, float sigma)
{
    return std::exp(-x*x/(2*sigma*sigma));
}

}} // namespace s3d::img

#endif
