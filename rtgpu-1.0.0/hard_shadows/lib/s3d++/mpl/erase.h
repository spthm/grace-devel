#ifndef S3D_MPL_ERASE_H
#define S3D_MPL_ERASE_H

#include "unref.h"
#include "vector_fwd.h"
#include "exists.h"
#include "add.h"
#include "../util/gcc.h"

namespace s3d { namespace mpl
{
	template <class VEC,
#if GCC_VERSION == 40500
			 int
#else
			 typename unref<VEC>::type::value_type
#endif
			 ...ITEMS>
	struct erase_c : erase_c<typename unref<VEC>::type, ITEMS...> {};

	template <class T, T...ITEMS>
	struct erase_c<vector_c<T>, ITEMS...>
	{
		typedef vector_c<T> type;
	};

	template <class T, T HEAD, T...TAIL, T...ITEMS>
	struct erase_c<vector_c<T,HEAD,TAIL...>, ITEMS...>
	{
		typedef typename std::conditional
		<
			exists_c<vector_c<T,ITEMS...>, HEAD>::value,
			erase_c<vector_c<T,TAIL...>, ITEMS...>,
			add<vector_c<T,HEAD>, erase_c<vector_c<T,TAIL...>, ITEMS...>>
		>::type::type type;
	};

	template <class VEC, class... ITEMS>
	struct erase : erase<typename unref<VEC>::type, ITEMS...> {};

	template <class...ITEMS>
	struct erase<vector<>, ITEMS...>
	{
		typedef vector<> type;
	};

	template <class HEAD, class...TAIL, class...ITEMS>
	struct erase<vector<HEAD,TAIL...>, ITEMS...>
	{
		typedef typename std::conditional
		<
			exists<vector<ITEMS...>, HEAD>::value,
			erase<vector<TAIL...>, ITEMS...>,
			add<vector<HEAD>, erase<vector<TAIL...>, ITEMS...>>
		>::type::type type;
	};

}} // namespace s3d::mpl

#endif
