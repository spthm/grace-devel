#ifndef S3D_MPL_EXISTS_H
#define S3D_MPL_EXISTS_H

#include <type_traits>
#include "unref.h"
#include "../util/gcc.h"

namespace s3d { namespace mpl
{

	template <class VEC, 
#if GCC_VERSION == 40500
			 int
#else
			 typename unref<VEC>::type::value_type 
#endif
			ITEM> 
	struct exists_c : exists_c<typename unref<VEC>::type, ITEM> {};

	template <class T, T ITEM>
	struct exists_c<vector_c<T>, ITEM>
	{
		static const bool value = false;
	};

	template <class T, T HEAD, T ITEM>
	struct exists_c<vector_c<T,HEAD>, ITEM>
	{
		static const bool value = HEAD == ITEM;
	};

	template <class T, T HEAD, T...TAIL, T ITEM>
	struct exists_c<vector_c<T,HEAD, TAIL...>, ITEM>
	{
		static const bool value = HEAD == ITEM || 
								  exists_c<vector_c<T,TAIL...>,ITEM>::value;
	};

	template <class VEC, class ITEM>
	struct exists : exists<typename unref<VEC>::type, ITEM> {};

	template <class ITEM>
	struct exists<vector<>, ITEM>
	{
		static const bool value = false;
	};

	template <class HEAD, class...TAIL, class ITEM>
	struct exists<vector<HEAD, TAIL...>, ITEM>
	{
		static const bool value = std::is_same<HEAD,ITEM>::value || 
								  exists<vector<TAIL...>,ITEM>::value;
	};

}} // namespace s3d::mpl

#endif
