#include "pch.h"
#include "packed_image.h"
#include "cast.h"
#include "../color/luminance.h"

namespace s3d { namespace img {

inline int align(int x, int a)
{
	int t = x%4;
	return x+(t==0?0:a-t);
}

packed_image::packed_image(const r2::usize &s)/*{{{*/
	: base({s.w, std::ceil(s.h/4.0)}, align(s.w,4))
	, m_height(s.h)
{
}/*}}}*/
packed_image::packed_image(size_t w, size_t h)/*{{{*/
	: base({w, std::ceil(h/4.0)}, align(w,4))
	, m_height(h)
{
}/*}}}*/
packed_image::packed_image(const luminance &srcmap)/*{{{*/
	: base({srcmap.w(), std::ceil(srcmap.h()/4.0)}, align(srcmap.w(),4))
	, m_height(srcmap.h())
{
	v4f *dst = data();
	const color::luminance *src = srcmap.data();

	size_t off1 = srcmap.row_stride(),
		   off2 = off1<<1,
		   off3 = off1+off2,
		   off4 = off3+off1;

	for(size_t y=0; y<srcmap.h(); y+=4)
	{
		const color::luminance *srccol = src;
		v4f *dstcol = dst;
		for(int x=(int)srcmap.w(); x; --x, ++src)
		{
			float a[4] __attribute__((aligned(16))) = {(float)src[0],0,0,0};
			if(y+1 < srcmap.h())
			{
				a[1] = src[off1];
				if(y+2 < srcmap.h())
				{
					a[2] = src[off2];
					if(y+3 < srcmap.h())
						a[3] = src[off3];
				}
			}

			*dst++ = v4f(a);
		}
		src = srccol + off4;
		dst = dstcol + row_stride();
	}
}/*}}}*/

packed_image::packed_image()/*{{{*/
	: base()
	, m_height(0)
{
}/*}}}*/

image packed_image::do_conv_to_radiance() const
{
	return image_cast<image>(*this);
}

packed_image::operator luminance() const/*{{{*/
{
	luminance dstmap(w(), h());

	size_t off1 = dstmap.row_stride(),
		   off2 = off1<<1,
		   off3 = off1+off2;

	const v4f *src = data();
	color::luminance *dst = dstmap.data();

	for(size_t y=0; y<dstmap.h(); y+=4)
	{
		const v4f *srccol = src;
		color::luminance *dstcol = dst;
		for(int x=(int)dstmap.w(); x; --x, ++dst, ++src)
		{
			float a[4] __attribute__((aligned(16)));
			src->convert_to(a);
			dst[0] = a[0];
			if(y+1 < dstmap.h())
			{
				dst[off1] = a[1];
				if(y+2 < dstmap.h())
				{
					dst[off2] = a[2];
					if(y+3 < dstmap.h())
						dst[off3] = a[3];
				}
			}
		}
		src = srccol+row_stride();
		dst = dstcol+dstmap.row_stride()*4;
	}

	return std::move(dstmap);
}/*}}}*/
packed_image &packed_image::operator=(packed_image &&that)/*{{{*/
{
	base::operator=(std::move(that));
	m_height = that.m_height;
	return *this;
}/*}}}*/
packed_image &packed_image::operator=(const packed_image &that)/*{{{*/
{
	base::operator=(that);
	m_height = that.m_height;
	return *this;
}/*}}}*/

packed_image transpose(const packed_image &srcimg)/*{{{*/
{
	packed_image dstimg(srcimg.h(), srcimg.w());

	const v4f *src = srcimg.data();
	v4f *dst = dstimg.data();

	for(int i=srcimg.packed_height(); i; --i)
	{
		v4f *dstcol = dst;

		for(int j=dstimg.packed_height(); j; --j, src+=4, dst+=dstimg.row_stride())
		{
			std::copy(src, src+4, dst);
			_MM_TRANSPOSE4_PS(dst[0], dst[1], dst[2], dst[3]);
		}
		dst = dstcol+4;
	}

	return std::move(dstimg);
}/*}}}*/
packed_image double_scale(const packed_image &srcimg)/*{{{*/
{
#if defined(__SSE3__)
	packed_image dstimg(srcimg.w()*2, srcimg.h()*2);

	int last_rows = srcimg.h() % 4;

	const v4f *src = srcimg.data();
	v4f *dst = dstimg.data();

	int dy = dstimg.row_stride(),
		dx = 1,
		sy = srcimg.row_stride();

#if 0
	const_cast<v4f *>(src)[0] = v4f(1,2,3,4);
	const_cast<v4f *>(src)[1] = v4f(5,6,7,8);
	const_cast<v4f *>(src)[sy] = v4f(9,10,11,12);
	const_cast<v4f *>(src)[sy+1] = v4f(13,14,15,16);
#endif

	for(int y=srcimg.packed_height(); y; --y)
	{
		const v4f *srccol = src;
		v4f *dstcol = dst;

		for(int x=srcimg.w(); x; --x, dst+=2, ++src)
		{
			assert(dst-dstimg.data()+2 <= (int)dstimg.length());
			assert(src-srcimg.data()+1 <= (int)srcimg.length());

			/* Objetivo:

			   1   5			 1	     1+5
			   2   6		   1+2   1+5+2+6
			   3   7			 2       2+6
			   4   8		   2+3	 2+3+6+7
						->
			   9  13			 3       3+7
			   10 14		   3+4   3+4+7+8
			   11 15		     4       4+8
			   12 16           4+9  4+9+8+13
			*/

			v4f s0,s1;
			int sx = x > 1 ? 1 : 0; // if we're on last column...

			// Are we on the last row?
			if(y == 1)
			{
				switch(last_rows)
				{
				case 1:
					s0 = _mm_shuffle_ps(src[0],src[0],// 1 | 1 | 1 | 1
										_MM_SHUFFLE(0,0,0,0));
					s1 = _mm_shuffle_ps(src[sx],src[sx], // 5 | 5 | 5 | 5
										_MM_SHUFFLE(0,0,0,0));
					break;
				case 2:
					s0 = _mm_shuffle_ps(src[0],src[0], // 1 | 2 | 2 | 2
										_MM_SHUFFLE(1,1,1,0));
					s1 = _mm_shuffle_ps(src[sx],src[sx], // 5 | 6 | 6 | 6
										_MM_SHUFFLE(1,1,1,0));
					break;
				case 3:
					s0 = _mm_shuffle_ps(src[0],src[0], // 1 | 2 | 3 | 3
										_MM_SHUFFLE(2,2,1,0));
					s1 = _mm_shuffle_ps(src[sx],src[sx], // 5 | 6 | 7 | 7
										_MM_SHUFFLE(2,2,1,0));
					break;
				default:
					s0 = src[0];
					s1 = src[sx];
					break;
				}
			}
			else
			{
				s0 = src[0];
				s1 = src[sx];
			}

			v4f e = s0 + s1,			 // 1+5 | 2+6 | 3+7 | 4+8
				f = _mm_hadd_ps(s0,s1), // 1+2 | 3+4 | 5+6 | 7+8
				i = _mm_shuffle_ps(s0,s0,//  4  |  4  |  3  |  2
								   _MM_SHUFFLE(1,2,3,3)),
				j = _mm_shuffle_ps(e, e,		 // 4+8 | 4+8 | 3+7 | 2+6
								   _MM_SHUFFLE(1,2,3,3));

			// Aren't we on last row?
			if(y > 1)
			{
				i = _mm_move_ss(i, src[sy]); //  9   |  4  |  3  |  2
				v4f g = src[sy]+src[sy+sx];	 // 9+13 | 10+14 | 11+15 | 12+16
				j = _mm_move_ss(j, g);		 // 9+13 | 4+8 | 3+7 | 2+6
			}

			v4f m = _mm_hadd_ps(j,i),		// 9+13+4+8 | 3+7+2+6 | 9+4 | 3+2
				n = _mm_hadd_ps(e,e);		// 1+5+2+6  | 3+7+4+8 | 1+5+2+6 |...

			f = _mm_shuffle_ps(f, m,		// 1+2 | 3+4 | 9+4  3+2
							   _MM_SHUFFLE(3,2,1,0));
			f = _mm_shuffle_ps(f, f,		// 1+2 | 3+2 | 3+4 | 9+4
							   _MM_SHUFFLE(2,1,3,0));

			m = _mm_movelh_ps(n, m);   // 1+5+2+6 | 3+7+4+8 | 9+13+4+8 | 3+7+2+6
			m = _mm_shuffle_ps(m, m,   // 1+5+2+6 | 3+7+2+6 | 3+7+4+8 | 9+13+4+8
							   _MM_SHUFFLE(2,1,3,0));

			e *= 0.5;
			f *= 0.5;
			m *= 0.25;

			dst[0] = _mm_unpacklo_ps(s0,  f);
			dst[dx] = _mm_unpacklo_ps(e, m);

			if(size_t(dst-dstimg.data()+dy+dx) < dstimg.length())
			{
				dst[dy] = _mm_unpackhi_ps(s0, f);
				dst[dy+dx] = _mm_unpackhi_ps(e, m);
			}
		}
		dst = dstcol + 2*dy;
		src = srccol + sy;
	}

	return std::move(dstimg);
#else
	throw std::runtime_error("Must have SSE3 extension enabled");
#endif
}/*}}}*/
packed_image half_scale(const packed_image &srcimg)/*{{{*/
{
	packed_image dstimg(std::ceil(srcimg.w()/2.0),std::ceil(srcimg.h()/2.0));

	const v4f *src = srcimg.data();
	v4f *dst = dstimg.data();

	int ss = srcimg.row_stride();

	int adjY = srcimg.packed_height()&1;

	for(int y=dstimg.packed_height()-adjY; y; --y)
	{
		const v4f *srcbeg = src;
		v4f *dstbeg = dst;
		for(int x=dstimg.w(); x; --x, src+=2)
			*dst++ = _mm_shuffle_ps(src[0], src[ss], _MM_SHUFFLE(2,0,2,0));

		dst = dstbeg + dstimg.row_stride();
		src = srcbeg+2*srcimg.row_stride();
	}

	if(adjY)
	{
		for(int x=dstimg.w(); x; --x, src+=2)
			*dst++ = src[0];
	}

	return std::move(dstimg);
}/*}}}*/

float maximum_difference(const packed_image &map)/*{{{*/
{
	v4f min_value(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX),
		max_value(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

	const v4f *data = map.data();

	int colstride = 4-(map.packed_height()*4 - map.h());
	if(colstride == 4)
		colstride = 0;

	for(int i=map.packed_height()-(colstride==0?0:1); i; --i)
	{
		const v4f *begcol = data;

		for(int j=map.w(); j; --j)
		{
			min_value = _mm_min_ps(*data, min_value);
			max_value = _mm_max_ps(*data, max_value);
			++data;
		}
		data = begcol + map.row_stride();
	}

	if(colstride)
	{
		v4f rowmask;
		switch(colstride)
		{
		case 1:
			rowmask = v4f(1,0,0,0) > v4f(0,0,0,0);
			break;
		case 2:
			rowmask = v4f(1,1,0,0) > v4f(0,0,0,0);
			break;
		case 3:
			rowmask = v4f(1,1,1,0) > v4f(0,0,0,0);
			break;
		}

		for(int j=map.w(); j; --j)
		{
			min_value = (_mm_min_ps(*data, min_value) & rowmask) 
							| andnot(min_value, rowmask);
			max_value = (_mm_max_ps(*data, max_value) & rowmask) 
							| andnot(max_value, rowmask);
		}
	}

	v4f d = max_value-min_value;
	float v[4] __attribute__((aligned(16)));
	d.convert_to(v);

	using std::max;

	float maxdiff = max(v[0], max(v[1], max(v[2], v[3])));
	return maxdiff;
}/*}}}*/

packed_image upsample(const packed_image &srcimg)/*{{{*/
{
	packed_image dstimg(srcimg.w()<<1, srcimg.h()<<1);

	const v4f *src = srcimg.data();
	v4f *dst = dstimg.data();

	v4f zero = 0.0;

	int dy = dstimg.row_stride();

	for(int i=srcimg.packed_height(); i; --i)
	{
		v4f *dstcol = dst;

		for(int j=srcimg.w(); j; --j, ++src, dst+=2)
		{
			dst[0] = _mm_unpacklo_ps(*src,zero);
			dst[dy] = _mm_unpackhi_ps(*src,zero);
			dst[1] = 0.0;
			dst[dy+1] = 0.0;
		}
		dst = dstcol+(dy<<1);
	}

	return std::move(dstimg);
}/*}}}*/
packed_image downsample(const packed_image &srcimg)/*{{{*/
{
	packed_image dstimg(srcimg.w()>>1, srcimg.h()>>1);

	const v4f *src = srcimg.data();
	v4f *dst = dstimg.data();

	int dy = dstimg.row_stride();

	for(int i=dstimg.packed_height(); i; --i)
	{
		const v4f *srccol = src;
		v4f *dstcol = dst;

		int sy = i==1 ? 0 : srcimg.row_stride();

		for(int j=dstimg.w(); j; --j, src+=2)
		{
			assert(src < srccol + srcimg.w());
			assert(src + sy < srcimg.data() + srcimg.length());
			*dst++ = _mm_shuffle_ps(src[0], src[sy], _MM_SHUFFLE(2,0,2,0));
		}

		src = srccol + sy*2;
		dst = dstcol + dy;
	}

	return std::move(dstimg);
}/*}}}*/

#if 0

packed_image copy(const packed_image &orig, const r2::ibox &bxorig)/*{{{*/
{
	packed_image img(bxorig.size);
	copy_into(img, orig, bxorig);
	return std::move(img);
}/*}}}*/

void copy_into(packed_image &dest, const image &orig)/*{{{*/
{
	copy_into(dest, r2::ibox(r2::iorigin, dest.size()),
			  orig, r2::ibox(r2::iorigin, orig.size()));
}/*}}}*/
void copy_into(packed_image &dest, const r2::ibox &bxdest, const packed_image &orig)/*{{{*/
{
	copy_into(dest, bxdest, orig, r2::ibox(r2::iorigin, orig.size()));
}/*}}}*/
void copy_into(packed_image &dest, const packed_image &orig, const r2::ibox &bxorig)/*{{{*/
{
	copy_into(dest, r2::ibox(r2::iorigin, dest.size()), orig, bxorig);
}/*}}}*/

void merge_line(v4f *dst, const v4f *src, int w, int a)/*{{{*/
{
	switch(a)
	{
	case 1:
		// s0 d1 d2 d3
		for(int j=w; j; --j, ++dst)
			*dst = _mm_move_ss(*src++, *dst);
		break;
	case 2:
		// s0 s1 d2 d3
		for(int j=w; j; --j, ++dst)
			*dst = _mm_shuffle_ps(*src++, *dst, _MM_SHUFFLE(3,2,1,0));
		break;
	case 3:
		// s0 s1 s2 d3
		for(int j=w; j; --j, ++dst)
		{
			*dst = _mm_shuffle_ps(*src, _mm_unpackhi_ps(*src, *dst),
								  _MM_SHUFFLE(3,0,1,0));
		}
	default:
		// s0 s1 s2 s3
		std::copy(src, src+rcdest.w, dst);
		break;
	}
}/*}}}*/

void copy_into(packed_image &dest, const r2::ibox &_rcdest, /*{{{*/
			   const packed_image &orig, const r2::ibox &_rcorig)
{
	r2::ibox rcdest = _rcdest & r2::ibox(r2::iorigin, dest.size()),
			 rcorig = _rcorig & r2::ibox(r2::iorigin, orig.size());

	if(rcdest.is_zero() || rcorig.is_zero())
		return;

	rcdest &= r2::ibox(rcdest.x, rcdest.y, rcorig.w, rcorig.h);

	assert(rcdest.size() == rcorig.size());

	int ro = rcorig.y & 3, // % 4
		rd = rcdest.y & 3; // % 4

	int dy = dest.row_stride(),
		sy = orig.row_stride();

	v4f *dst = &dest.pixel_at(rcdest.x, rcdest.y/4);
	const v4f *src = &orig.pixel_at(rcorig.x, rc.orig.y/4);

	if(ro == rd)
	{
		// Processa a primeira linha
		merge_line(dst, src, rcdest.w, ro);
		src += sy;
		dst += dy;

		// Processa o meio
		for(int i=rcdest.h-2; i; --i)
		{
			std::copy(src, src+rcdest.w, dst);
			src += sy;
			dst += dy;
		}

		// Processa a última linha
		merge_line(dst, src, rcdest.w, (rcorig.y + rcorig.h) & 3);
	}
	else




}/*}}}*/
#endif

}}
