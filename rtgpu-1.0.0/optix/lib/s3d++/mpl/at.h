#ifndef S3D_MPL_AT_H
#define S3D_MPL_AT_H

#include "unref.h"
#include "vector_fwd.h"

namespace s3d { namespace mpl
{
	template <int N, class T>
	struct at : at<N, typename unref<T>::type> {};

	template <int N, class T, T HEAD, T...TAIL>
	struct at<N, vector_c<T,HEAD,TAIL...>> : at<N-1, vector_c<T,TAIL...>>
	{
	};

	template <class T, T HEAD, T...TAIL>
	struct at<0, vector_c<T,HEAD,TAIL...>>
	{
		static const T value = HEAD;
	};

	template <int N, class HEAD, class...TAIL>
	struct at<N, vector<HEAD,TAIL...>> : at<N-1, vector<TAIL...>>
	{
	};

	template <class HEAD, class...TAIL>
	struct at<0, vector<HEAD,TAIL...>>
	{
		typedef HEAD type;
	};
}} // namespace s3d::mpl

#endif
