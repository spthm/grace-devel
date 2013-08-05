#ifndef S3D_MPL_ADD_H
#define S3D_MPL_ADD_H

#include "unref.h"
#include "vector.h"

namespace s3d { namespace mpl
{

	template <class... ARGS> 
	struct add : add<typename unref<ARGS>::type...> {};

	template <class T, T...II> 
	struct add<vector_c<T,II...>>
	{
		typedef vector_c<T,II...> type;
	};

	template <class T, T...II, T...JJ, class... ARGS> 
	struct add<vector_c<T,II...>, vector_c<T,JJ...>, ARGS...>
	{
		typedef typename add<vector_c<T,II..., JJ...>, ARGS...>::type type;
	};

	template <class...TT> 
	struct add<vector<TT...>>
	{
		typedef vector<TT...> type;
	};

	template <class...TT, class...UU, class... ARGS> 
	struct add<vector<TT...>, vector<UU...>, ARGS...>
	{
		typedef typename add<vector<TT..., UU...>, ARGS...>::type type;
	};

}} // namespace s3d::mpl

#endif
