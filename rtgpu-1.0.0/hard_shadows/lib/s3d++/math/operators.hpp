#include "../util/type_traits.h"

namespace s3d { namespace math
{

namespace detail
{
	template <class T, class U>
	struct default_result_add
	{
		static const int diff = order<T>::value - order<U>::value;

		static const int D1 = diff==0 ? order<T>::value : (diff>0 ? diff : 0),
						 D2 = diff==0 ? order<U>::value : (diff<0 ? -diff : 0);

		typedef typename rebind<typename std::conditional<(diff>=0),T,U>::type,
				decltype(std::declval<typename s3d::value_type<T,D1>::type>()+
						 std::declval<typename s3d::value_type<U,D2>::type>()),
				diff==0 && order<T>::value==0 ? 0 : 1>::type type;
	};

	template <class T, class U>
	struct default_result_sub
	{
		static const int diff = order<T>::value - order<U>::value;

		static const int D1 = diff==0 ? order<T>::value : (diff>0 ? diff : 0),
						 D2 = diff==0 ? order<U>::value : (diff<0 ? -diff : 0);

		typedef typename rebind<typename std::conditional<(diff>=0),T,U>::type,
				decltype(std::declval<typename s3d::value_type<T,D1>::type>()-
						 std::declval<typename s3d::value_type<U,D2>::type>()),
				diff==0 && order<T>::value==0 ? 0 : 1>::type type;
	};

	template <class T, class U>
	struct default_result_mul
	{
		static const int diff = order<T>::value - order<U>::value;

		static const int D1 = diff==0 ? order<T>::value : (diff>0 ? diff : 0),
						 D2 = diff==0 ? order<T>::value : (diff<0 ? -diff : 0);

		typedef typename rebind<typename std::conditional<(diff>=0),T,U>::type,
				decltype(std::declval<typename s3d::value_type<T,D1>::type>()*
						 std::declval<typename s3d::value_type<U,D2>::type>()),
				diff==0 && order<T>::value==0 ? 0 : 1>::type type;
	};

	template <class T, class U>
	struct default_result_div
	{
		static const int diff = order<T>::value - order<U>::value;

		static const int D1 = diff==0 ? order<T>::value : (diff>0 ? diff : 0),
						 D2 = diff==0 ? order<U>::value : (diff<0 ? -diff : 0);

		typedef typename rebind<typename std::conditional<(diff>=0),T,U>::type,
				decltype(std::declval<typename s3d::value_type<T,D1>::type>()/
						 std::declval<typename s3d::value_type<U,D2>::type>()),
				diff==0 && order<T>::value==0 ? 0 : 1>::type type;
	};

	template <class R>
	struct is_result_specialized
	{
		template <class X>
		static mpl::bool_<sizeof(typename X::type)?true:false> impl(int);

		template <class X>
		static mpl::bool_<false> impl(...);

		typedef decltype(impl<R>(0)) ret_type;

		static const bool value = ret_type::value;
	};

	template <template<class,class> class R,
			  template<class,class> class DR, class T, class U, class EN=void>
	struct process_concept_and_get_result
	{
		typedef typename std::conditional
		<
			is_result_specialized<R<T,U>>::value,
			R<T,U>,
			DR<typename remove_concept<T>::type, 
			   typename remove_concept<U>::type>
		>::type type;
	};

	template <template<class,class> class R,
			  template<class,class> class DR, class T, class U>
	struct process_concept_and_get_result<R,DR,T,U,
		typename std::enable_if<is_concept<T>::value && 
							    !is_concept<U>::value>::type>
	{
		typedef typename std::conditional
		<
			is_result_specialized<R<T,U>>::value,
			std::identity<R<T,U>>,
			process_concept_and_get_result<R,DR,
						T,typename concept_arg<U>::type>
		>::type::type type;
	};

	template <template<class,class> class R,
			  template<class,class> class DR, class T, class U>
	struct process_concept_and_get_result<R,DR,T,U,
		typename std::enable_if<!is_concept<T>::value && 
							    is_concept<U>::value>::type>
	{
		typedef typename std::conditional
		<
			is_result_specialized<R<T,U>>::value,
			std::identity<R<T,U>>,
			process_concept_and_get_result<R,DR,
				typename concept_arg<T>::type, typename remove_concept<U>::type>
		>::type::type type;
	};

	template <template<class,class> class R,
			  template<class,class> class DR, class T, class U, class EN=void>
	struct process_result
	{
		typedef typename process_concept_and_get_result
		<
			R,DR,
			T, typename concept_arg<U>::type
		>::type type;
	};

	template <template<class,class> class R,
			  template<class,class> class DR, class T, class U>
	struct process_result<R,DR,T,U,
		typename std::enable_if<is_result_specialized<R<T,U>>::value>::type>
	{
		typedef R<T,U> type;
	};

	// ---------------------------------------------------

	template <class R, class NEXT=R, class EN=void> struct check_valid
	{
	};
	template <template<class,class> class R, class T, class U, class NEXT>
	struct check_valid<R<T,U>, NEXT, typename std::enable_if<
		std::is_base_of<operators, T>::value ||
		std::is_base_of<operators, U>::value
		>::type>
	{
		typedef NEXT type;
	};

	template <class R, class NEXT=R, class EN=void> 
	struct check_dont_have_overload
	{
		typedef NEXT type;
	};
	template <class R, class NEXT> 
	struct check_dont_have_overload<R, NEXT, typename std::enable_if<
		sizeof(R::result_type::call(std::declval<typename R::lhs_type>(),
								    std::declval<typename R::rhs_type>()))
		>::type>
	{
	};

	template <class R, class NEXT=R, class EN=void> 
	struct check_have_overload
	{
		typedef NEXT type;
	};

	template <class R, class NEXT> 
	struct check_have_overload<R, NEXT, typename std::enable_if<
		sizeof(typename check_dont_have_overload<R>::type)>::type>
	{
	};

	template <class R, bool B, class NEXT=R, class EN=void> struct check_direct;
	template <template<class,class> class R,class T,class U,bool B, class NEXT> 
	struct check_direct<R<T,U>, B, NEXT, typename std::enable_if<
		(order<T>::value >= order<U>::value)==B>::type>
	{
		typedef NEXT type;
	};

	template <class R, class NEXT=R, class EN=void>
	struct check_dont_revert
	{
		typedef NEXT type;
	};
	template <class R, class NEXT> 
	struct check_dont_revert<R, NEXT, typename std::enable_if<
					R::result_type::revert>::type>
	{
	};

	template <class R, class NEXT=R, class EN=void>
	struct check_revert
	{
	};
	template <class R, class NEXT> 
	struct check_revert<R, NEXT, typename std::enable_if<
					R::result_type::revert>::type>
	{
		typedef NEXT type;
	};

	template <class R, class EN=void> struct get_direct_result;
	template <class R>
	struct get_direct_result<R, typename std::enable_if<
		sizeof(typename 
		check_valid<R,
		check_dont_have_overload<R,
		check_direct<R,true,
		check_dont_revert<R>>>>::type::type::type::type)>::type>
	{
		typedef typename R::type type;
	};

	template <class R, class EN=void> struct get_reverse_result;
	template <class R>
	struct get_reverse_result<R, typename std::enable_if<
		sizeof(typename 
		check_valid<R,
		check_dont_have_overload<R,
		check_revert<R>>>::type::type::type)>::type>
	{
		typedef typename R::type type;
	};

	template <class R>
	struct get_reverse_result<R, typename std::enable_if<
		sizeof(typename 
		check_valid<R,
		check_dont_have_overload<R,
		check_direct<R,false>>>::type::type::type)>::type>
	{
		typedef typename R::type type;
	};

	template <class R, class EN=void> struct get_overload_result;
	template <class R>
	struct get_overload_result<R, typename std::enable_if<
		sizeof(typename check_valid<R,
			check_have_overload<R>>::type::type)>::type>
	{
		typedef typename R::type type;
	};
}

template <class T, class U>
struct result_add_dispatch/*{{{*/
{
	typedef T lhs_type;
	typedef U rhs_type;

	typedef typename detail::process_result
	<
		math::result_add,
		detail::default_result_add,
		T, U
	>::type result_type;

	typedef typename result_type::type type;
};/*}}}*/

template <class T, class U>
struct result_sub_dispatch/*{{{*/
{
	typedef T lhs_type;
	typedef U rhs_type;

	typedef typename detail::process_result
	<
		math::result_sub,
		detail::default_result_sub,
		T, U
	>::type result_type;

	typedef typename result_type::type type;
};/*}}}*/

template <class T, class U>
struct result_mul_dispatch/*{{{*/
{
	typedef T lhs_type;
	typedef U rhs_type;

	typedef typename detail::process_result
	<
		math::result_mul,
		detail::default_result_mul,
		T, U
	>::type result_type;

	typedef typename result_type::type type;
};/*}}}*/

template <class T, class U>
struct result_div_dispatch/*{{{*/
{
	typedef T lhs_type;
	typedef U rhs_type;

	typedef typename detail::process_result
	<
		math::result_div,
		detail::default_result_div,
		T, U
	>::type result_type;

	typedef typename result_type::type type;
};/*}}}*/


// addition
template <class T, class U>
inline auto operator+(const T &u, const U &v)
	-> typename detail::get_direct_result<result_add_dispatch<T,U>>::type
{
	return typename result_add_dispatch<T, U>::type(u) += v;
}

template <class T, class U>
inline auto operator+(const T &u, const U &v)
	-> typename detail::get_reverse_result<result_add_dispatch<T,U>>::type
{
	return typename result_add_dispatch<T,U>::type(v) += u;
}

template <class T, class U>
typename detail::get_overload_result<result_add_dispatch<T,U>>::type 
operator+(const T &u, const U &v) __attribute__((always_inline));

template <class T, class U>
inline auto operator+(const T &u, const U &v)
	-> typename detail::get_overload_result<result_add_dispatch<T,U>>::type
{
	return result_add_dispatch<T,U>::result_type::call(u,v);
}

// multiplication

template <class T, class U>
inline auto operator*(const T &u, const U &v)
	-> typename detail::get_direct_result<result_mul_dispatch<T,U>>::type
{
	return typename result_mul_dispatch<T, U>::type(u) *= v;
}

template <class T, class U>
inline auto operator*(const T &u, const U &v)
	-> typename detail::get_reverse_result<result_mul_dispatch<T,U>>::type
{
	return typename result_mul_dispatch<T, U>::type(v) *= u;
}

template <class T, class U>
typename detail::get_overload_result<result_mul_dispatch<T,U>>::type
operator*(const T &u, const U &v) __attribute__((always_inline));

template <class T, class U>
inline auto operator*(const T &u, const U &v)
	-> typename detail::get_overload_result<result_mul_dispatch<T,U>>::type
{
	return result_mul_dispatch<T,U>::result_type::call(u,v);
}

// subtraction

template <class T, class U>
inline auto operator-(const T &u, const U &v)
	-> typename detail::get_direct_result<result_sub_dispatch<T,U>>::type
{
	return typename result_sub_dispatch<T, U>::type(u) -= v;
}

template <class T, class U>
inline auto operator-(const T &u, const U &v)
	-> typename detail::get_reverse_result<result_sub_dispatch<T,U>>::type
{
	return typename result_sub_dispatch<T, U>::type(-v) += u;
}

template <class T, class U>
typename detail::get_overload_result<result_sub_dispatch<T,U>>::type
operator-(const T &u, const U &v) __attribute__((always_inline));

template <class T, class U>
inline auto operator-(const T &u, const U &v)
	-> typename detail::get_overload_result<result_sub_dispatch<T,U>>::type
{
	return result_sub_dispatch<T,U>::result_type::call(u,v);
}

// division 

template <class T, class U>
inline auto operator/(const T &u, const U &v)
	-> typename detail::get_direct_result<result_div_dispatch<T,U>>::type
{
	return typename result_div_dispatch<T, U>::type(u) /= v;
}

template <class T, class U>
inline auto operator/(const T &u, const U &v)
	-> typename detail::get_reverse_result<result_div_dispatch<T,U>>::type
{
	return typename result_div_dispatch<T, U>::type(inv(v)) *= u;
}

template <class T, class U>
typename detail::get_overload_result<result_div_dispatch<T,U>>::type
operator/(const T &u, const U &v) __attribute__((always_inline));

template <class T, class U>
inline auto operator/(const T &u, const U &v)
	-> typename detail::get_overload_result<result_div_dispatch<T,U>>::type
{
	return result_div_dispatch<T,U>::result_type::call(u,v);
}

}} // namespace s3d::math
