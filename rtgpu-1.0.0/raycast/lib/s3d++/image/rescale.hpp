#include "../math/r2/point.h"
#include "../math/r2/box.h"

namespace s3d { namespace img
{

template <class P>
Image<P> rescale(const r2::isize &szdest, const Image<P> &orig, const r2::ibox &bxorig)/*{{{*/
{
	Image<P> img(szdest);
	rescale_into(img, orig, bxorig);
	return std::move(img);
}/*}}}*/
template <class P>
Image<P> rescale(const r2::isize &szdest, const Image<P> &orig)/*{{{*/
{
	Image<P> img(szdest);
	rescale_into(img, orig);
	return std::move(img);
}/*}}}*/

template <class P>
void rescale_into(Image<P> &dest, const Image<P> &orig)/*{{{*/
{
	rescale_into(dest, r2::ibox(r2::iorigin, dest.size()),
				orig, r2::ibox(r2::iorigin, orig.size()));
}/*}}}*/
template <class P>
void rescale_into(Image<P> &dest, const r2::ibox &bxdest, const Image<P> &orig)/*{{{*/
{
	rescale_into(dest, bxdest, orig, r2::ibox(r2::iorigin, orig.size()));
}/*}}}*/
template <class P>
void rescale_into(Image<P> &dest, const Image<P> &orig, const r2::ibox &bxorig)/*{{{*/
{
	rescale_into(dest, r2::ibox(r2::iorigin, dest.size()), orig, bxorig);
}/*}}}*/
template <class P>
void rescale_into(Image<P> &dest, const r2::ibox &_rcdest, /*{{{*/
				 const Image<P> &orig, const r2::ibox &_rcorig)
{
	r2::ibox rcdest = _rcdest & r2::ibox(r2::iorigin, dest.size()),
			 rcorig = _rcorig & r2::ibox(r2::iorigin, orig.size());

	if(rcdest.is_zero() || rcorig.is_zero())
		return;
	else if(rcorig.size == rcdest.size)
		copy_into(dest, rcdest, orig, rcorig);
	else
	{
		real ax = rcorig.w/real(rcdest.w),
			 ay = rcorig.h/real(rcdest.h);

		for(int i=rcdest.y; i<rcdest.y+rcdest.h; ++i)
		{
			for(int j=rcdest.x; j<rcdest.x+rcdest.w; ++j)
			{
				dest[i][j] = bilinear_pixel(orig, {(j-rcdest.x)*ax+rcorig.x, 
												   (i-rcdest.y)*ay+rcorig.y});
			}
		}
	}
}/*}}}*/

template <class T> Image<T> downsample(const Image<T> &srcimg)/*{{{*/
{
	return half_scale(srcimg);
}/*}}}*/
template <class T> Image<T> upsample(const Image<T> &srcimg)/*{{{*/
{
	Image<T> dstimg(srcimg.w()*2, srcimg.h()*2);
	dstimg.clear();

	const T *src = srcimg.data();
	T *dst = dstimg.data();

	int drow = dstimg.row_stride();

	for(size_t y=0; y<srcimg.h(); ++y)
	{
		for(size_t x=0; x<srcimg.w(); ++x, dst+=2, ++src)
			*dst = *src;

		dst += drow;
	}

	return std::move(dstimg);
}/*}}}*/

template <class T> 
Image<T> double_scale(const Image<T> &srcimg)/*{{{*/
{
	Image<T> dstimg(srcimg.w()*2, srcimg.h()*2);

	const T *src = srcimg.data();
	T *dst = dstimg.data();

	int drow = dstimg.row_stride(),
		srow = srcimg.row_stride();

	for(int y=0; y<srcimg.h(); ++y)
	{
		for(int x=0; x<srcimg.w(); ++x, dst+=2, ++src)
		{
			dst[0] = src[0];
			if(x+1 < srcimg.w())
				dst[1] = (src[0]+src[1])/2;
			else
				dst[1] = src[0];

			if(y+1 < srcimg.h())
			{
				dst[drow] = (src[0]+src[srow])/2;
				if(x+1 < srcimg.w())
					dst[drow+1] = (src[0] + src[1] + src[srow] + src[srow+1])/4;
				else
					dst[drow+1] = dst[drow];
			}
			else
			{
				dst[drow]   = dst[0];
				dst[drow+1] = dst[1];
			}
		}
		dst += drow;
	}

	return std::move(dstimg);
}/*}}}*/

template <class T> 
Image<T> half_scale(const Image<T> &srcimg)/*{{{*/
{
	using std::ceil;

	Image<T> dstimg(ceil(srcimg.w()/2.0), ceil(srcimg.h()/2.0));

	const T *src = srcimg.data();
	T *dst = dstimg.data();

	for(size_t y=0; y<srcimg.h(); y+=2)
	{
		for(size_t x=0; x<srcimg.w(); x+=2)
		{
			*dst++ = *src++;
			if(x+1 < srcimg.w())
				++src;
		}

		if(y+1 < srcimg.w())
			src += srcimg.row_stride();
	}
	return std::move(dstimg);
}/*}}}*/


}} // namespace s3d::img
