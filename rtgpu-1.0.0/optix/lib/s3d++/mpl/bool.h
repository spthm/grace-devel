#ifndef S3D_MPL_BOOL_H
#define S3D_MPL_BOOL_H

#include "integral.h"

namespace s3d { namespace mpl
{
	template <bool B> 
	struct bool_ : integral<bool, B> 
	{
	};
}} // namespace s3d::mpl

#endif
