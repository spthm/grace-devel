#ifndef S3D_MPL_INTEGRAL_H
#define S3D_MPL_INTEGRAL_H

namespace s3d { namespace mpl
{
	template <class T, T N> 
	struct integral
	{
		static const T value = N;
		typedef T type;
	};

}} // namespace s3d::mpl

#endif
