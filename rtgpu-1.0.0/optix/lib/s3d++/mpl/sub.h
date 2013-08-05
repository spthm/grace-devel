#ifndef S3D_MPL_SUB_H
#define S3D_MPL_SUB_H

#include <type_traits>
#include "unref.h"
#include "vector_fwd.h"
#include "exists.h"
#include "add.h"

namespace s3d { namespace mpl
{

	template <class... ARGS> 
	struct sub : sub<typename unref<ARGS>::type...> {};

	// some trivial shortcuts

	template <class T, T...II>
	struct sub<vector_c<T,II...>>
	{
		typedef vector_c<T,II...> type;
	};

	template <class T, class... ARGS>
	struct sub<vector_c<T>, ARGS...>
	{
		typedef vector_c<T> type;
	};

	template <class... T>
	struct sub<vector<T...>>
	{
		typedef vector<T...> type;
	};

	template <class... ARGS>
	struct sub<vector<>, ARGS...>
	{
		typedef vector<> type;
	};

	// main processing
	template <class T, T I, T...II,  class... ARGS> 
	struct sub<vector_c<T,I,II...>, ARGS...>
	{
		typedef typename std::conditional
		<
			exists_c<add<ARGS...>,I>::value,
			sub<vector_c<T,II...>, ARGS...>,
			add<vector_c<T,I>, sub<vector_c<T,II...>, ARGS...>>
		>::type::type type;
	};

	template <class T, class...TT,  class... ARGS> 
	struct sub<vector<T,TT...>, ARGS...>
	{
		typedef typename std::conditional
		<
			exists<add<ARGS...>,T>::value,
			sub<vector<TT...>, ARGS...>,
			add<vector<T>, sub<vector<TT...>, ARGS...>>
		>::type::type type;
	};


}} // namespace s3d::mpl

#endif
