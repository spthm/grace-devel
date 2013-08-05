#ifndef S3D_IMAGE_RESCALE_H
#define S3D_IMAGE_RESCALE_H

#include "fwd.h"
#include "../math/r2/fwd.h"

namespace s3d { namespace img
{

template <class P> 
Image<P> rescale(const r2::isize &szdest, const Image<P> &orig, 
			 const r2::ibox &bxorig);

template <class P>
Image<P> rescale(const r2::isize &szdest, const Image<P> &orig);

template <class P>
void rescale_into(Image<P> &dest, const Image<P> &orig);

template <class P>
void rescale_into(Image<P> &dest, const r2::ibox &bxdest, const Image<P> &orig);

template <class P>
void rescale_into(Image<P> &dest, const Image<P> &orig, const r2::ibox &bxorig);

template <class P>
void rescale_into(Image<P> &dest, const r2::ibox &bxdest, 
				  const Image<P> &orig, const r2::ibox &bxorig);

void rescale_into(Image<> &dest, const r2::ibox &bxdest, 
				  const Image<> &orig, const r2::ibox &bxorig);

template <class T> Image<T> double_scale(const Image<T> &srcimg);
template <class T> Image<T> half_scale(const Image<T> &srcimg);
template <class T> Image<T> transpose(const Image<T> &srcimg);

template <class T> Image<T> downsample(const Image<T> &srcimg);
template <class T> Image<T> upsample(const Image<T> &srcimg);

}} // namespace s3d::img

#include "rescale.hpp"

#endif
