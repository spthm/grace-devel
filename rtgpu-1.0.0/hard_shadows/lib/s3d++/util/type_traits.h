/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License 
	version 3 as published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public 
	License along with S3D++. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef S3D_UTIL_TYPE_TRAITS_H
#define S3D_UTIL_TYPE_TRAITS_H

//#include <boost/mpl/print.hpp>
#include "type_traits_fwd.h"
#include <type_traits>
#include <iosfwd>
#include <cstdint>
#include <memory>
#include "../mpl/int.h"
#include "../mpl/bool.h"
#include "gcc.h"
#include "tuple.h"
#include "concepts.h"

#if GCC_VERSION < 40500
namespace std
{
	template <class T>
	typename add_rvalue_reference<T>::type declval();
}
#endif

namespace s3d
{

template <class T>
struct print
{
//	typedef typename boost::mpl::print<T>::type type;
};


// is_tuple ---------------------------------------

template <class T>
struct is_tuple : std::false_type
{
};

template <class...T>
struct is_tuple<std::tuple<T...>> : std::true_type
{
};

// copy_const/volatile/cv ---------------------------------------

template <class FROM, class TO>
struct copy_const
{
	typedef TO type;
};
template <class FROM, class TO>
struct copy_const<const FROM, TO>
{
	typedef const TO type;
};

template <class FROM, class TO>
struct copy_const<const FROM &, TO>
{
	typedef const TO type;
};


template <class FROM, class TO>
struct copy_volatile
{
	typedef TO type;
};
template <class FROM, class TO>
struct copy_volatile<volatile FROM, TO>
{
	typedef volatile TO type;
};
template <class FROM, class TO>
struct copy_volatile<volatile FROM &, TO>
{
	typedef volatile TO type;
};

template <class FROM, class TO>
struct copy_cv
{
	typedef typename copy_const
	<
		FROM,
		typename copy_volatile<FROM,TO>::type
	>::type type;
};

// remove_ref_cv/cv_ref --------------------------------------

template <class T>
struct remove_ref_cv
{
	typedef typename std::remove_reference
	<
		typename std::remove_cv<T>::type
	>::type type;
};

template <class T>
struct remove_cv_ref
{
	typedef typename std::remove_cv
	<
		typename std::remove_reference<T>::type
	>::type type;
};

// has_ostream_operator/cv_ref --------------------------------------

namespace detail/*{{{*/
{
	// trick attributed to David Abrahams

	struct no_match {};
	struct matchall
	{
		template <class T>
		matchall(T &&);
	};
	no_match operator<<(const matchall &, const matchall &);

	template <class T>
	struct has_ostream_operator
	{
	public:
		static const bool value = !std::is_same
		<
//			decltype(std::declval<std::ostream>() << std::declval<const T>()),
			decltype(*(std::ostream *)NULL << *(const T *)NULL),
			no_match	
		>::value;
	};
}/*}}}*/
using detail::has_ostream_operator;


// difference_type --------------------------------------------------

template <class T>
struct difference_type
{
	typedef decltype(std::declval<T>() - std::declval<T>()) type;
};

// value_type --------------------------------------------------


namespace detail
{
	template <class T> struct has_value_type {};

	struct null_base {};

	template <class T, int N>
	auto test_vtype(const T &, const mpl::int_<N> &)
		-> typename std::enable_if<sizeof(typename T::value_type), 
				value_type<has_value_type<T>,N>>::type;

	null_base test_vtype(...);

	template <class T, int N>
	auto test_vtype(const T &, const mpl::int_<N> &)
		-> typename std::enable_if<is_concept<T>::value,
			value_type<typename remove_concept<T>::type,N>>::type;
};


template <class T, int N>
struct value_type<detail::has_value_type<T>,N> 
	: value_type<typename T::value_type,N-1>
{
};

template <class T>
struct value_type<T, 0>
{
	typedef typename remove_concept<T>::type type;
};

template <class T>
struct value_type<T&, 0> : value_type<T,0>
{
};

template <class T>
struct value_type<detail::has_value_type<T>, 0> : value_type<T,0>
{
};

template <class T>
struct value_type<T*,0>
{
	typedef T *type;
};

template <class T, int N>
struct value_type<T*,N> : value_type<T,N-1>
{
};

template <class T, int N>
struct value_type<T &,N> : value_type<T,N>
{
};

template <class T, int N>
struct value_type<std::shared_ptr<T> ,N> : value_type<T,N-1>
{
};

template <class T, int N>
struct value_type<std::weak_ptr<T> ,N> : value_type<T,N-1>
{
};

template <class T, int N>
struct value_type<std::unique_ptr<T> ,N> : value_type<T,N-1>
{
};

template <class T, int N>
struct value_type 
	: std::remove_reference<
		 decltype(detail::test_vtype(std::declval<T>(),mpl::int_<N>()))>::type
{
};

template <template<class,int...> class V, class T, int... II, int N>
struct value_type<V<T,II...>,N>
	: std::remove_reference<
		 decltype(detail::test_vtype(std::declval<V<T,II...>>(),mpl::int_<N>()))>::type
{
};

template <class T, int N>
struct value_type<std::reference_wrapper<T>,N> : value_type<T,N-1>
{
};

template <template<class,int...> class V, class T, int... II>
struct value_type<V<T,II...>,0> 
{
	typedef typename remove_concept<V<T,II...>>::type type;
};

// order --------------------------------------------------------

template <class T>
struct order;

namespace detail
{
	template <class T>
	auto order_impl(T &&)
		-> mpl::int_<order<typename value_type<T>::type>::value + 1>;

	mpl::int_<0> order_impl(...);
}

template <class T>
struct order
{
	typedef decltype(detail::order_impl(std::declval<T>())) ret_type;

	static const int value = ret_type::value;
};

template <class T>
struct order<T *>
{
	static const int value = order<T>::value + 1;
};

template <class T>
struct order<T &> : order<typename std::remove_cv<T>::type>
{
};

// safe_value_ype --------------------------------------------------------

template <class T, int L=1>
struct safe_value_type
{
	typedef typename value_type<T,
			(order<T>::value < L ? order<T>::value : L)>::type type;
};


// max_index --------------------------------------------------------

namespace detail/*{{{*/
{
	inline std::pair<int,int> max_index()
	{
		return std::make_pair(-1,0);
	}

	template <class T, class...ARGS>
	std::pair<int,T> max_index(const T &a, const ARGS &...rest)
	{
		std::pair<int,T> aux = max_index(rest...);
		if(aux.second > a)
			return std::make_pair(aux.first+1, aux.second);
		else
			return std::make_pair(0,a);
	}
}/*}}}*/

template <class T, class...ARGS>
int max_index(const T &first, const ARGS &...rest)
{
	return detail::max_index(first, rest...).first;
}

template <class T> 
struct higher_precision;

template <> struct higher_precision<int8_t>/*{{{*/
{
	typedef int16_t type;
};/*}}}*/
template <> struct higher_precision<int16_t>/*{{{*/
{
	typedef int32_t type;
};/*}}}*/
template <> struct higher_precision<int32_t>/*{{{*/
{
	typedef int64_t type;
};/*}}}*/
template <> struct higher_precision<uint8_t>/*{{{*/
{
	typedef uint16_t type;
};/*}}}*/
template <> struct higher_precision<uint16_t>/*{{{*/
{
	typedef uint32_t type;
};/*}}}*/
template <> struct higher_precision<uint32_t>/*{{{*/
{
	typedef uint64_t type;
};/*}}}*/

template <> struct higher_precision<float>/*{{{*/
{
	typedef double type;
};/*}}}*/
template <> struct higher_precision<double>/*{{{*/
{
	typedef long double type;
};/*}}}*/

// make_signed/unsigned ----------------------------------

namespace detail
{
	template <class T, bool FLOAT = std::is_floating_point<T>::value>
	struct __make_signed
	{
		typedef T type;
	};

	template <class T>
	struct __make_signed<T,false>
	{
		typedef typename std::make_signed<T>::type type;
	};

	template <class T, bool FLOAT = std::is_floating_point<T>::value>
	struct __make_unsigned
	{
		typedef T type;
	};

	template <class T>
	struct __make_unsigned<T,false>
	{
		typedef typename std::make_unsigned<T>::type type;
	};
}

template <class T>
struct make_signed
{
	typedef typename detail::__make_signed<T>::type type;
};

template <class T>
struct make_unsigned
{
	typedef typename detail::__make_unsigned<T>::type type;
};

namespace detail
{
	template <class FROM, class TO, class EN = void>
	struct is_explicitly_convertible : std::false_type
	{
	};

	template <class FROM, class TO>
	struct is_explicitly_convertible<FROM,TO,
		typename std::enable_if<sizeof(decltype(TO(std::declval<const FROM>())))>::type>
			: std::true_type
	{
	};
}

template <class FROM, class TO>
struct is_explicitly_convertible
{
	static const bool value = detail::is_explicitly_convertible<FROM,TO>::value;;
};


// rebind ------------------------------------------------

namespace detail
{
	template <class T, class U, int N>
	struct rebind_impl
	{
		typedef typename rebind_impl
		<
			T, 
			typename rebind_impl
			<
				typename value_type<T>::type,
				U,
				N-1
			>::type,
			1
		>::type type;
	};

	template <class T, class U>
	typename std::enable_if
	<
		is_concept<T>::value, 
		rebind<typename T::type,U,1>
	>::type rebind_choose(int);

	template <class T, class U>
	typename T::template rebind<U> rebind_choose(int,...);

	template <class T, class U>
	void rebind_choose(...);

	template <class T, class U>
	struct rebind_impl<T,U,1>
	{
		typedef decltype(rebind_choose<T,U>(0)) ret_type;

		static_assert(!std::is_same<ret_type,void>::value, 
					  "You must define rebind for this type");

		typedef typename ret_type::type type;
	};

	template <class T, class U, int N>
	struct rebind_impl<T*,U, N>
	{
		typedef typename rebind_impl<T,U,N-1>::type *type;
	};

	// this is very common throughout the library
	template <template <class,int...> class V, class T, class U, int...II>
	struct rebind_impl<V<T,II...>,U,1>
	{
		typedef typename std::conditional
		<
			is_concept<V<T,II...>>::value,
			rebind_impl<typename remove_concept<V<T,II...>>::type, U, 1>,
			std::identity<V<U,II...>>
		>::type::type type;
	};

	template <class T, class U>
	struct rebind_impl<T,U,0>
	{
		typedef U type;
	};

	template <class T, class U, int N>
	mpl::bool_<sizeof(typename rebind<T,U,N>::type)?true:false> 
		has_rebind_impl(const T &, const U &, const mpl::int_<N> &);

	mpl::bool_<false> has_rebind_impl(...);
}

template <class T, class U, int N>
struct rebind : detail::rebind_impl<T,U, N==-1 ? order<T>::value : N>
{
};

template <class T, class U, int N>
struct has_rebind
{
	typedef decltype(detail::has_rebind_impl(T(), U(), mpl::int_<N>())) aux;

	static const bool value = aux::value;
};

// this is very common throughout the library
template <template <class,int...> class V, class T, class U, int...D> 
struct rebind<V<T,D...>, U>/*{{{*/
{
	typedef V<U,D...> type;
}; /*}}}*/

// make_floating_point ------------------------------------------------

template <class T>
struct make_floating_point
{
	typedef typename value_type<T,order<T>::value>::type deep_type;

	typedef typename std::conditional
	<
		std::is_floating_point<deep_type>::value,
		std::identity<T>,
		rebind<T,float>
	>::type::type type;
};

// is_pointer_like ------------------------------------------------------

template <class T>
struct is_pointer_like : std::is_pointer<T>
{
};

template <class T>
struct is_pointer_like<const T> : is_pointer_like<T>
{
};

template <class T>
struct is_pointer_like<std::shared_ptr<T>> : std::true_type
{
};

template <class T>
struct is_pointer_like<std::unique_ptr<T>> : std::true_type
{
};

template <class T>
struct is_pointer_like<T&> : is_pointer_like<T>
{
};

static_assert(is_pointer_like<std::shared_ptr<int>&>::value, "ERRO");
static_assert(is_pointer_like<const std::shared_ptr<int>&>::value, "ERRO");
static_assert(is_pointer_like<std::shared_ptr<int>>::value, "ERRO");
static_assert(is_pointer_like<int*>::value, "ERRO");
static_assert(is_pointer_like<int*&>::value, "ERRO");

// is_pointer_like ------------------------------------------------------

template <class T>
struct is_reference_wrapper : std::false_type
{
};

template <class T>
struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type
{
};

template <class T>
struct is_reference_wrapper<T&> : is_reference_wrapper<T>
{
};

template <class T>
struct is_reference_wrapper<const T> : is_reference_wrapper<T>
{
};

} // namespace s3d

#endif

// $Id: type_traits.h 3054 2010-08-28 23:31:06Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

