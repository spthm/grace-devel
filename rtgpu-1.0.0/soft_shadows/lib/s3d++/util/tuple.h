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

#ifndef S3D_UTIL_TUPLE_H
#define S3D_UTIL_TUPLE_H

#include <tuple>
#include <functional>
#include "../mpl/create_range.h"

namespace s3d
{

// It's rather sad that we have to resort to this in order to extract
// the first element of the tuple and return the rest...

namespace detail
{
	template <class A, class...ARGS, int ...INDEXES>
	std::tuple<ARGS...> get_tail_helper(const std::tuple<A, ARGS...> &t,
										const mpl::vector_c<int,INDEXES...> &)
	{
		return std::tuple<ARGS...>(std::get<(INDEXES+1)>(t)...);
	}
}

template <class A, class...ARGS> 
A get_head(const std::tuple<A,ARGS...> &t)
{
	return std::get<0>(t);
}

template <class A> 
std::tuple<> get_tail(const std::tuple<A> &t)
{
	return std::tuple<>();
}

template <class A, class...ARGS> 
std::tuple<ARGS...> get_tail(const std::tuple<A, ARGS...> &t)
{
	return detail::get_tail_helper(t, typename mpl::create_range<int,0,sizeof...(ARGS)>::type());
}


// Returns the index (0-based) in tuple T of type V, -1 if not found

namespace detail
{
	template <int S, class V, class T>
	struct find_helper;

	template <int S, class V, class A, class... ARGS>
	struct find_helper<S, V, std::tuple<A,ARGS...>>
	{
		static const int value = find_helper<S, V, std::tuple<ARGS...>>::value;
	};

	template <int S, class V, class... ARGS>
	struct find_helper<S, V, std::tuple<V,ARGS...>>
	{
		static const int value = S - sizeof...(ARGS)-1;
	};

	template <int S, class V>
	struct find_helper<S, V, std::tuple<>>
	{
		static const int value = -1; 
	};
}

template <class V, class T>
struct find;

template <class V, class... ARGS>
struct find<V, std::tuple<ARGS...>>
{
	static const int value = detail::find_helper<sizeof...(ARGS), V,
												 std::tuple<ARGS...>>::value;
};

// Returns whether 'VAL' exists in tuple T during run time

template <class DUMMY=void>
inline bool exists_in(int i)
{
	return false;
}

template <int HEAD, int... TAIL>
bool exists_in(int i)
{
	if(HEAD == i)
		return true;
	else
		return exists_in<TAIL...>(i);
}

namespace detail
{
	template <class F, class...ARGS, int...II>
	auto call_helper(F fn, const std::tuple<ARGS...>&args, 
					 mpl::vector_c<int, II...>)
		-> typename std::result_of<F(ARGS...)>::type
	{
		return fn(std::get<II>(args)...);
	}
}

template <class F, class...ARGS>
auto call(F fn, const std::tuple<ARGS...> &args)
	-> typename std::result_of<F(ARGS...)>::type
{
	return detail::call_helper(fn, args, 
				typename mpl::create_range<int,0,sizeof...(ARGS)>::type());
}

}

#endif

// $Id: tuple.h 2959 2010-08-13 19:20:39Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

