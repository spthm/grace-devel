#ifndef S3D_MATH_TRAITS_H
#define S3D_MATH_TRAITS_H

#include "../mpl/exists.h"

namespace s3d { namespace math
{

enum coords_kind
{
	HOMOGENEOUS,
	HETEROGENEOUS
};

namespace traits
{
	// dimension --------------------------------
	template <class C> struct dim
	{
		static_assert(std::is_arithmetic<C>::value, 
					  "You must partially specialize this class for your type");

		typedef mpl::vector_c<int,1> type;
	};

	template <template <class,int...> class V, class T, int...D> 
	struct dim<V<T,D...>>/*{{{*/
	{
		typedef mpl::vector_c<int,D...> type;
	}; /*}}}*/

	template <template <class> class V, class T>
	struct dim<V<T>>/*{{{*/
	{
		typedef mpl::vector_c<int,V<T>::dim> type;
	};/*}}}*/

	// coords_kind -------------------------------------------

	template <class T, class EN = void>
	struct coords_kind
	{
		static const math::coords_kind value = HETEROGENEOUS;
	};

	template <class T>
	struct coords_kind<T, typename std::enable_if<sizeof(std::declval<T>().begin())>::type>/*{{{*/
	{
		static const math::coords_kind value = HOMOGENEOUS;
	};/*}}}*/
}

template <class C> struct is_static
{
	static const bool value = !mpl::exists_c<traits::dim<C>,RUNTIME>::value;
};
template <class C> struct is_runtime
{
	static const bool value = !is_static<C>::value;
};

// is_view ----------------------------------------------

template <class C> struct is_view;

namespace detail
{
	template <class C, class EN=void>
	struct is_view_impl
	{
		static const bool value = false;
	};

	template <class C>
	struct is_view_impl<C, 
		typename std::enable_if<sizeof(typename C::space_type)>::type>
		: is_view<typename C::space_type>
	{
	};
}

// must be partially specialized for view spaces

template <class C> 
struct is_view : detail::is_view_impl<C>/*{{{*/
{
};/*}}}*/



}} // namespace s3d::math::traits

#endif
