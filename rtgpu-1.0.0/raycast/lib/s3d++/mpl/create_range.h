#ifndef S3D_MPL_CREATE_RANGE_H
#define S3D_MPL_CREATE_RANGE_H

#include "vector_fwd.h"

namespace s3d { namespace mpl
{
	template <class T, T BEG, T END, T INCR=1, class VEC=vector_c<T>>
	struct create_range;

	template <class T, T CUR, T END, T INCR, T... ITEMS> 
	struct create_range<T, CUR, END, INCR, vector_c<T,ITEMS...>>
	{
		typedef typename std::conditional
		<
			(CUR < END),
			create_range<T, CUR+INCR, END, INCR, vector_c<T,ITEMS...,CUR>>,
			std::identity<vector_c<T,ITEMS...>>
		>::type::type type;
	};
}} // namespace s3d::mpl

#endif
