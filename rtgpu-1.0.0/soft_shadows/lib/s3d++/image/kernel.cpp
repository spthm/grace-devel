#include "pch.h"
#include "kernel.h"

namespace s3d { namespace img {

kernel gaussian_kernel(float sigma, int dim)
{

	if(dim == 0)
		dim = static_cast<int>(2*4*sigma);

	// It isn't odd?
	if((dim & 1) == 0)
		++dim; // must have a central value

	kernel k(dim);


	float sum = 0;
	int middle = k.size()/2;

	for(int i=0; i<(int)k.size(); ++i)
	{
		int relpos = i-middle;
		k[i] = gaussian_weight(relpos, sigma);
		sum += k[i];
	}

	std::transform(k.begin(), k.end(), k.begin(),
				  std::bind2nd(std::divides<float>(), sum));

	return std::move(k);
}

kernel blur_kernel(real a)
{
	return {real(0.25)-a/2, real(0.25), a, real(0.25), real(0.25)-a/2};
}


}} // namespace s3d::img
