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

#ifndef S3D_MEM_FN_H
#define S3D_MEM_FN_H

namespace s3d
{

namespace detail
{
	template <class T, class U, class R, class... ARGS> 
	class mem_fn_caller_t/*{{{*/
	{
	public:
		mem_fn_caller_t(T *ptr, std::function<R(U*,ARGS...)> member)
			: m_ptr(ptr), m_member(member) {}

		R operator()(ARGS&&... args)
		{
			return m_member(m_ptr, std::forward<ARGS>(args)...);
		}

	private:
		T *m_ptr;
		std::function<R(T*,ARGS...)> m_member;
	};/*}}}*/
}

template <class T, class U, class R, class... ARGS> 
auto mem_fn(R(T::*member)(ARGS...), U *obj)
	-> detail::mem_fn_caller_t<T,U,R,ARGS...> 
{
	return {obj, member};
}

template <class T, class U, class R, class... ARGS> 
auto mem_fn(R(T::*member)(ARGS...) const, const U *obj)
	-> detail::mem_fn_caller_t<const T, const U,R,ARGS...> 
{
	return {obj, member};
}

template <class T, class U, class R, class... ARGS> 
auto mem_fn(R(T::*member)(ARGS...) volatile, volatile U *obj)
	-> detail::mem_fn_caller_t<volatile T, volatile U,R,ARGS...> 
{
	return {obj, member};
}


template <class T, class U, class R, class... ARGS> 
auto mem_fn(R(T::*member)(ARGS...) const volatile, const volatile U *obj)
	-> detail::mem_fn_caller_t<const volatile T, const volatile U,R,ARGS...> 
{
	return {obj, member};
}

} // namespace s3d

#endif

// $Id: mem_fn.h 2892 2010-07-30 00:22:28Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

