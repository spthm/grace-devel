#include "pch.h"
#include "image.h"
#include "../color/luminance.h"
#include "../color/radiance.h"
#include "packed_image.h"
#include "convolution.h"

namespace s3d { namespace img {

grayscale_pyramid laplacian_pyramid(luminance &&img, int levels)/*{{{*/
{
	grayscale_pyramid pyr;
	kernel K = blur_kernel();

	for(int level=0; level < levels && img.w()>1 && img.h()>1; ++level)
	{
		luminance blur = convolve(img, K);

		pyr.push_back(img-blur);

		img = downsample(blur);
	}

	pyr.push_back(std::move(img));

	reverse(pyr.begin(), pyr.end());

	return std::move(pyr);
}/*}}}*/
grayscale_pyramid laplacian_pyramid(const luminance &img, int levels)/*{{{*/
{
	return laplacian_pyramid(luminance(img), levels);
}/*}}}*/
luminance laplacian_reconstruct(const grayscale_pyramid &pyr)/*{{{*/
{
	kernel K = blur_kernel();

	luminance img = pyr[0];

	for(unsigned i=1; i<pyr.size(); ++i)
		img = pyr[i] + convolve(upsample(4*img), K);

	return std::move(img);
}/*}}}*/

color_pyramid laplacian_pyramid(const image &img, int levels)/*{{{*/
{
	luminance comp[3];
	decompose(img, comp[0], comp[1], comp[2]);

	color_pyramid pyramid;

	for(int i=0; i<3; ++i)
		pyramid[i] = laplacian_pyramid(std::move(comp[i]), levels);

	return std::move(pyramid);
}/*}}}*/
image laplacian_reconstruct(const color_pyramid &pyr)/*{{{*/
{
	luminance comp[3];

	for(int i=0; i<3; ++i)
		comp[i] = laplacian_reconstruct(pyr[i]);

	return compose<color::radiance>(comp[0], comp[1], comp[2]);
}/*}}}*/

}}
