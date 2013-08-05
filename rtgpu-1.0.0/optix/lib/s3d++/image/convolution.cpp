#include "pch.h"
#include "image.h"
#include "convolution.h"
#include "kernel.h"
#include "../math/v4f.h"
#include "../color/luminance.h"
#include "packed_image.h"

#define USE_ASSEMBLY 1

namespace s3d { namespace img {

template <class T> T *border(T *x, T *beg, T *end)/*{{{*/
{
	assert(end-beg >= 2);

	if(x < beg)
		return border(beg + (beg-x), beg, end);
	else if(x >= end)
		return border(end - (x-end)-2, beg, end);
	return x;
}/*}}}*/

// luminance

color::luminance convolve_value(const color::luminance *beg, /*{{{*/
								const color::luminance *src, 
								const color::luminance *end, 
								const color::luminance *min, 
								const color::luminance *max, const kernel &K)
{
	auto *curmask = &K[0], *endmask = curmask + K.size();

	real dst = 0;

	while(src < min)
		dst += *border(src++, min, max) * *curmask++;

	end = src + std::min(endmask-curmask, max-src);

	while(src != end)
		dst += *src++ * *curmask++;

	while(curmask < endmask)
		dst += *border(src++, min, max) * *curmask++;

	return color::luminance(dst);
}/*}}}*/
luminance convolve_transpose_1d(const luminance &srcimg, const r2::ibox &area, const kernel &K)/*{{{*/
{
	luminance dstimg(transpose(srcimg.size()));

	const auto *minsrc = srcimg.data(),
			   *maxsrc = minsrc + srcimg.w(),
			   *begsrc = minsrc + area.x,
			   *endsrc = begsrc + area.w,
			   *src = minsrc;

	auto *dstcol = dstimg.data();

	int kmiddle = K.size()/2;

	for(int i=0; i<(int)srcimg.h(); ++i)
	{
		color::luminance *dst = dstcol;

		for(int j=0; j<(int)srcimg.w(); ++j, ++src, dst += dstimg.row_stride())
		{
			if(i < area.y || i >= area.y+area.h ||
			   j < area.x || j >= area.x+area.w)
			{
				*dst = *src;
			}
			else
			{
				*dst = convolve_value(begsrc, src-kmiddle, endsrc, 
									  minsrc, maxsrc, K);
			}
		}

		begsrc += srcimg.row_stride();
		endsrc += srcimg.row_stride();
		minsrc += srcimg.row_stride();
		maxsrc += srcimg.row_stride();
		++dstcol;
	}
	return std::move(dstimg);
}/*}}}*/

luminance convolve(const luminance &src, const r2::ibox &area, const kernel &K)/*{{{*/
{
	return convolve_transpose_1d(
					convolve_transpose_1d(src, area, K), 
					transpose(area), K);
}/*}}}*/
luminance convolve(const luminance &src, const kernel &K)/*{{{*/
{
	return convolve(src, r2::ibox(0,0,src.w(), src.h()), K);
}/*}}}*/

// packed_image

typedef std::vector<v4f> packed_kernel;

inline __m128 convolve_value(const v4f *minsrc, const v4f *src, const v4f *maxsrc, const packed_kernel &K)/*{{{*/
{
	const v4f *curmask = &K[0],
			  *endmask = curmask+K.size();

	__m128 dst = _mm_setzero_ps();

	while(src < minsrc)
		dst = dst + *border(src++, minsrc, maxsrc) * *curmask++;

	const v4f *endsrc = src + std::min(endmask-curmask, maxsrc-src);

	if(src >= endsrc)
		return dst;

	// O compilador sisma de colocar [dst] na pilha e ficar atualizando-o
	// lá a cada iteração do loop (3 acessos à memoria, no total).
	// O codigo abaixo faz a coisa certa, ou seja, 2 acessos a memoria.
	// PS: esse eh o loop mais custoso da convolucao
#if USE_ASSEMBLY
	// Nao tah dando p/ fazer o somatorio direto em dst, o compilador
	// 'pensa' que seu valor estah em xmm1, mas nao estah.
	__m128 aux;
	asm volatile(
		"   xorps  %[dst], %[dst] \n"
		"0: \n"
		"	movaps (%[src]), %%xmm0 \n"
		"	add    $16, %[src] \n"
		"	mulps  (%[K]), %%xmm0 \n"
		"	add    $16, %[K] \n"
		"	addps  %%xmm0, %[dst] \n"
		"	cmp    %[src], %[endsrc] \n"
		"	jne    0b \n"
		: [dst]"=x"(aux), [src]"+r"(src), [K]"+r"(curmask)
		: [endsrc]"r"(endsrc)
		: "xmm0", "cc");
	dst = dst + aux;
#else
	while(src != endsrc)
		dst = dst + *src++ * *curmask++;
#endif

	while(curmask < endmask)
		dst = dst + *border(src++, minsrc, maxsrc) * *curmask++;

	return dst;
}/*}}}*/
packed_image convolve_transpose_1d(const packed_image &srcimg, const packed_kernel &K)/*{{{*/
{
	packed_image dstimg(transpose(srcimg.size()));

	const v4f *src = srcimg.data(),
			  *minsrc = src,
			  *maxsrc = src+srcimg.w();

	int kmiddle = K.size()/2;

	v4f *dstcol = dstimg.data();

	for(int i=srcimg.packed_height(); i; --i)
	{
		v4f *dst = dstcol;

		for(int j=dstimg.packed_height(); j; --j, dst+=dstimg.row_stride())
		{
			assert(size_t(src-srcimg.data())+4 <= srcimg.length());

			__m128 a = convolve_value(minsrc, src++ - kmiddle, maxsrc, K),
				   b = convolve_value(minsrc, src++ - kmiddle, maxsrc, K),
				   c = convolve_value(minsrc, src++ - kmiddle, maxsrc, K),
				   d = convolve_value(minsrc, src++ - kmiddle, maxsrc, K);

			_MM_TRANSPOSE4_PS(a, b, c, d);

			assert(size_t(dst-dstimg.data())+4 <= dstimg.length());
			dst[0] = a;
			dst[1] = b;
			dst[2] = c;
			dst[3] = d;
		}
		minsrc += srcimg.row_stride();
		maxsrc += srcimg.row_stride();
		dstcol+=4;
	}

	return std::move(dstimg);
}/*}}}*/

packed_image convolve(const packed_image &src, const kernel &_K)/*{{{*/
{
	packed_kernel K(_K.size());
	copy(_K.begin(), _K.end(), K.begin());

	return convolve_transpose_1d(convolve_transpose_1d(src, K), K);
}/*}}}*/

}} // namespace s3d::img
