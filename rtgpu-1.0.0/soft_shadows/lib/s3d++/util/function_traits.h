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

#ifndef S3D_UTIL_FUNCTION_TRAITS_HPP
#define S3D_UTIL_FUNCTION_TRAITS_HPP

#include <type_traits>
#include <functional>
#include "arity.h"
#include "any.h"
#include "../mpl/create_range.h"

namespace s3d
{

template <class F>
struct function_traits/*{{{*/
{
	typedef typename function_traits<decltype(&F::operator())>::result result;
	typedef typename function_traits<decltype(&F::operator())>::args args;
	static const int arity = std::tuple_size<args>::value;

	typedef typename function_traits<decltype(&F::operator())>::sig sig;
};/*}}}*/

template <class R, class...ARGS> 
struct function_traits<R(ARGS...)>/*{{{*/
{
	typedef R result;
	typedef std::tuple<ARGS...> args;
	static const int arity = sizeof...(ARGS);

	typedef R sig(ARGS...);
};/*}}}*/

template <class R, class...ARGS> 
struct function_traits<R(&)(ARGS...)>/*{{{*/
{
	typedef R result;
	typedef std::tuple<ARGS...> args;
	static const int arity = sizeof...(ARGS);

	typedef R sig(ARGS...);
};/*}}}*/

template <class R, class...ARGS> 
struct function_traits<R(*)(ARGS...)>/*{{{*/
{
	typedef R result;
	typedef std::tuple<ARGS...> args;
	static const int arity = sizeof...(ARGS);

	typedef R sig(ARGS...);
};/*}}}*/

template <class R, class T, class...ARGS> 
struct function_traits<std::_Mem_fn<R (T::*)(ARGS...)> >/*{{{*/
{
	typedef R result;
	typedef std::tuple<T*,ARGS...> args;
	static const int arity = sizeof...(ARGS)+1;

	typedef R sig(T*, ARGS...);
};/*}}}*/

template <class R, class T, class...ARGS> 
struct function_traits<std::_Mem_fn<R (T::*)(ARGS...) const> >/*{{{*/
{
	typedef R result;
	typedef std::tuple<const T *, ARGS...> args;
	static const int arity = sizeof...(ARGS)+1;

	typedef R sig(const T*, ARGS...);
};/*}}}*/

template <class R, class T, class...ARGS> 
struct function_traits<std::_Mem_fn<R (T::*)(ARGS...) volatile> >/*{{{*/
{
	typedef R result;
	typedef std::tuple<volatile T *, ARGS...> args;
	static const int arity = sizeof...(ARGS)+1;

	typedef R sig(volatile T *, ARGS...);
};/*}}}*/

template <class R, class T, class...ARGS> 
struct function_traits<std::_Mem_fn<R (T::*)(ARGS...) const volatile> >/*{{{*/
{
	typedef R result;
	typedef std::tuple<const volatile T*, ARGS...> args;
	static const int arity = sizeof...(ARGS)+1;

	typedef R sig(const volatile T*, ARGS...);
};/*}}}*/

template <class F, class...ARGS> 
struct function_traits<std::_Bind<F(ARGS...)> >/*{{{*/
{
	/* This deserves some explanation:
	
	   std::bind is made of a callable object and some bound arguments.
	   Those bound arguments may contain placeholders so that the user
	   can specify at call time what gets in what parameter. Non-placeholder
	   arguments are specified at bind time.

	   The template parameter 'F' is the callable object type, 'ARGS...' are
	   the bound arguments.

	   We want to retrieve the call signature of the whole bind expression,
	   performing placeholder substitutions as they appear.

	   Example:

	   R foo(A,B,C,D) {}
	   auto bar = bind(&foo, _1, _2, C(), _3);
	   assert(is_same<decltype(bar), R(A,B,D)>::value == true);

	   The plan to get R(A,B,D) out of F(ARGS...) is:

	   - Calculate the arity of the resulting callable object (3 in the example
	     above)
	   - Iterate over each result argument (I)
	   - For each result argument, find its position in the bound argument
	     list (J).
	   - If the result argument is being used in the bound argument list,
	     get the type of the corresponding argument of 'F'. This is the type
	     of the result argument.
	   - If it isn't in the bound argument list, the result argument type is
	     s3d::any, since it will be discarded (not used in 'F').

	   Of course, we have to deduce the return type too, but this is easy
	   (std::result_of)
    */

	// Bound arguments as a tuple
	typedef std::tuple<ARGS...> TARGS;

	static const int arity = s3d::arity<std::_Bind<F(ARGS...)> >::value;

	// Iterates over bound arguments (J), testing whether bound argument 'J' is
	// a placeholder for destination argument 'I'.
	template <int I, int J>
	struct find_match
	{
		typedef typename std::conditional
		<
			// Is bound argument 'J' a placeholder for destination argument
			// 'I' (placeholders numbering is 1-based, hence I+1
			std::is_placeholder
			<
				typename std::tuple_element<J, TARGS>::type
			>::value == I+1,

			// Yes, we've found a match. Returns the corresponding argument
			// type in the callable object.
			typename std::tuple_element
			<
				J,
				typename function_traits<F>::args
			>::type,

			// Nope, test the next bound argument
			typename find_match<I, J+1>::type
		>::type type;
	};

	// End of bound arguments and no match? Returns 'void', signaling
	// no match
	template <int I>
	struct find_match<I, sizeof...(ARGS)>
	{
		typedef void type;
	};


	// For each destination argument,
	template <int I>
	struct map_dest_arg
	{
		// Look for the corresponding type in F
		// (begin looking in the first bound argument (0))
		typedef typename find_match<I, 0>::type MATCH;

		typedef typename std::conditional
		<
			// Didn't found it (find_match returns 'void')
			std::is_same<MATCH, void>::value,
			any,  // returns any (will be discarded)
			MATCH // returns the type found
		>::type type;
	};


	// Iterates over each destination argument (II), 
	template <class II> struct calc_dest_args;
	template <int... II> struct calc_dest_args<mpl::vector_c<int,II...>>
	{
		// Returns the destination arguments in a tuple
		typedef std::tuple
		<
			typename map_dest_arg<II>::type...
		> type;
	};

	typedef typename calc_dest_args
	<
		typename mpl::create_range<int,0,arity>::type
	>::type args;


	// Given a return type and a tuple with argument types, return
	// a function call type.
	template <class T> struct signature_from_tuple;
	template <class... DARGS> 
	struct signature_from_tuple<std::tuple<DARGS...> >
	{
		typedef typename std::result_of<std::_Bind<F(ARGS...)>(DARGS...)>::type type(DARGS...);
	};

	typedef typename signature_from_tuple<args>::type sig;
};/*}}}*/

template <class T> 
struct function_traits<std::function<T>>/*{{{*/
{
	typedef typename function_traits<T>::result result;
	typedef typename function_traits<T>::args args;
	static const int arity = function_traits<T>::arity;

	typedef typename function_traits<T>::type sig;
};/*}}}*/

template <class R, class T, class...ARGS> 
struct function_traits<R(T::*)(ARGS...)>/*{{{*/
{
	typedef R result;
	typedef std::tuple<ARGS...> args;
	static const int arity = sizeof...(ARGS);

	typedef R sig(ARGS...);
};/*}}}*/

template <class R, class T, class...ARGS> 
struct function_traits<R(T::*)(ARGS...) const>/*{{{*/
{
	typedef R result;
	typedef std::tuple<ARGS...> args;
	static const int arity = sizeof...(ARGS);

	typedef R sig(ARGS...);
};/*}}}*/

template <class R, class T, class...ARGS> 
struct function_traits<R(T::*)(ARGS...) volatile>/*{{{*/
{
	typedef R result;
	typedef std::tuple<ARGS...> args;
	static const int arity = sizeof...(ARGS);

	typedef R sig(ARGS...);
};/*}}}*/

template <class R, class T, class...ARGS> 
struct function_traits<R(T::*)(ARGS...) const volatile>/*{{{*/
{
	typedef R result;
	typedef std::tuple<ARGS...> args;
	static const int arity = sizeof...(ARGS);

	typedef R sig(ARGS...);
};/*}}}*/

template <class T>
struct is_functor/*{{{*/
{
	static const bool value = std::is_function<T>::value ||
							  std::is_member_function_pointer<T>::value;
};/*}}}*/
template <class R, class T, class...ARGS>
struct is_functor<std::_Mem_fn<R (T::*)(ARGS...)> >/*{{{*/
{
	static const bool value = true;
};/*}}}*/
template <class R, class T, class...ARGS>
struct is_functor<std::_Mem_fn<R (T::*)(ARGS...) const> >/*{{{*/
{
	static const bool value = true;
};/*}}}*/
template <class R, class T, class...ARGS>
struct is_functor<std::_Mem_fn<R (T::*)(ARGS...) volatile> >/*{{{*/
{
	static const bool value = true;
};/*}}}*/
template <class R, class T, class...ARGS>
struct is_functor<std::_Mem_fn<R (T::*)(ARGS...) const volatile> >/*{{{*/
{
	static const bool value = true;
};/*}}}*/
template <class R, class...ARGS>
struct is_functor<std::function<R(ARGS...)> >/*{{{*/
{
	static const bool value = true;
};/*}}}*/
template <class F, class...ARGS>
struct is_functor<std::_Bind<F(ARGS...)> >/*{{{*/
{
	static const bool value = true;
};/*}}}*/

}

#endif

// $Id: function_traits.h 2959 2010-08-13 19:20:39Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

