#ifndef S3D_MPL_UNREF_H
#define S3D_MPL_UNREF_H

#include "vector.h"

namespace s3d { namespace mpl
{
	template <class T, class DUMMY=void>
	struct unref
	{
		static_assert(sizeof(T), "T must be defined");
		typedef T type;
	};

	template <class T>
	struct unref<T,typename std::enable_if<sizeof(typename T::type)>::type>
	{
		typedef typename T::type type;
	};

#if 0

	// little optimization for vectors so that we don't have to always include
	// vector.h 
	template <class...TT>
	struct unref<vector<TT...>>
	{
		typedef vector<TT...> type;
	};

	template <class T, T...II>
	struct unref<vector_c<T,II...>>
	{
		typedef vector_c<T, II...> type;
	};
#endif
}} // namespace s3d::mpl

#endif
