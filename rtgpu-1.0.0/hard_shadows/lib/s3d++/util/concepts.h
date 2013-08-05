#ifndef S3D_UTIL_CONCEPTS_H
#define S3D_UTIL_CONCEPTS_H

#include <type_traits>
#include "../mpl/bool_fwd.h"

namespace s3d
{
	namespace detail
	{
		// Default case, value is false, so result shortcircuits to false
		template <bool, class... B>
		struct and_ret
		{
			static const bool value = false;
			typedef void type;
		};

		// Value is true, with remaining values to check
		template <class B1, class... B>
		struct and_ret<true, B1, B...>
		{
			typedef and_ret<B1::value, B...> aux;

			static const bool value = aux::value;
			typedef typename aux::type type;
		};

		// Last value is true, the last parameter is a type (return type)
		template <class R>
		struct and_ret<true,R>
		{
			static const bool value = true;
			typedef R type;
		};
	}

	template <class C1, class...CONDS_RET>
	struct requires :
		std::enable_if<detail::and_ret<C1::value, CONDS_RET...>::value, 
			  typename detail::and_ret<C1::value, CONDS_RET...>::type>
	{
	};

	template <class T>
	struct concept
	{
		typedef T type;
		static const bool is_concept = true;
	};


	template <class T>
	struct arithmetic : concept<T> {};

	template <class T, class EN=void>
	struct concept_arg
	{
		typedef T type;
	};

	template <class T>
	struct concept_arg<T, 
		typename std::enable_if<std::is_arithmetic<T>::value>::type>
	{
		typedef arithmetic<T> type;
	};

	namespace detail
	{
		template <class T>
		mpl::bool_<T::is_concept> is_concept_impl(int);
		template <class T>
		mpl::bool_<false> is_concept_impl(...);
	};

	template <class T>
	struct is_concept
	{
		typedef decltype(detail::is_concept_impl<T>(0)) ret_type;

		static const bool value = ret_type::value;
	};

	template <class T, class EN>
	struct remove_concept
	{
		typedef T type;
	};

	template <class T>
	struct remove_concept<T,typename std::enable_if<is_concept<T>::value>::type>
	{
		typedef typename T::type type;
	};
}

#endif
