#ifndef S3D_MPL_UNIQUE_H
#define S3D_MPL_UNIQUE_H

#include "unref.h"
#include "vector_fwd.h"
#include "erase.h"

namespace s3d { namespace mpl
{

	template <class VEC> 
	struct unique : unique<typename unref<VEC>::type> {};

	template <class T> 
	struct unique<vector_c<T>>
	{
		typedef vector_c<T> type;
	};

	template <class T, T HEAD, T... TAIL> 
	struct unique<vector_c<T,HEAD,TAIL...>>
	{
		typedef typename add
		<
			std::identity<vector_c<T,HEAD>>,
			erase_c<unique<vector_c<T,TAIL...>>,HEAD>
		>::type::type type;
	};

	template <> 
	struct unique<vector<>>
	{
		typedef vector<> type;
	};

	template <class HEAD, class... TAIL> 
	struct unique<vector<HEAD,TAIL...>>
	{
		typedef typename add
		<
			std::identity<vector<HEAD>>,
			erase<unique<vector<TAIL...>>,HEAD>
		>::type::type type;
	};


}} // namespace s3d::mpl

#endif
