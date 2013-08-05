#ifndef S3D_MPL_INT_H
#define S3D_MPL_INT_H

#include "integral.h"

namespace s3d { namespace mpl
{
	template <int N> 
	struct int_ : integral<int, N> 
	{
	};
}} // namespace s3d::mpl

#endif
