/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License version 3 as 
	published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public License
	along with S3D++.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef S3D_UTIL_ARITY_H
#define S3D_UTIL_ARITY_H

#include <functional>

namespace s3d
{

namespace detail /*{{{*/
{
	template <class T, class enable=void> struct get_max_arity
	{
		static const int value = std::is_placeholder<T>::value;
	};

	template <int A, class...ARGS> struct calc_bind_arity;/*{{{*/

	template <int A, class ARG, class... ARGS> struct calc_bind_arity<A,ARG,ARGS...>
	{
		static const int value = calc_bind_arity<(get_max_arity<ARG>::value>A)?get_max_arity<ARG>::value:A, ARGS...>::value;
	};
	template <int A> struct calc_bind_arity<A>
	{
		static const int value = A;
	};/*}}}*/
}/*}}}*/

template <class T> struct arity;/*{{{*/
template <class R, class... ARGS> struct arity<R(ARGS...)>
{
	static const int value = sizeof...(ARGS);
};
template <class R, class... ARGS> struct arity<R(*)(ARGS...)>
{
	static const int value = sizeof...(ARGS);
};
template <class T, class R, class... ARGS> struct arity<R(T::*)(ARGS...)>
{
	static const int value = sizeof...(ARGS)+1; // +1 por causa do this
};
template <class T, class R, class... ARGS> struct arity<R(T::*)(ARGS...) const>
{
	static const int value = sizeof...(ARGS)+1; // +1 por causa do this
};
template <class F, class... ARGS> struct arity<std::_Bind<F(ARGS...)> >
{
	static const int value = detail::calc_bind_arity<0,ARGS...>::value;
};
template <class T> struct arity<std::_Mem_fn<T>>
{
	static const int value = arity<T>::value;
};
template <class T> struct arity<std::function<T>>
{
	static const int value = arity<T>::value;
};/*}}}*/

} // namespace s3d

#endif

// $Id: arity.h 2373 2009-06-29 02:26:26Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

