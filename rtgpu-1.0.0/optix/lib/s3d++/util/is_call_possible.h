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

#ifndef S3D_UTIL_IS_CALL_POSSIBLE_H
#define S3D_UTIL_IS_CALL_POSSIBLE_H

#include "arity.h"

namespace s3d
{

// This mess basically enables us to check whether a function (or functor)
// call is valid at compile time, mainly for use with SFINAE idiom
//
// is_call_possible<F, SIG>::value == true/false
//
// Example:
// is_call_possible<double(int), void(double)>::value == true 
//      one can call a function, i.e. double func(int x) like this: func(2.0),
//      ignoring the return value

template <class TYPE, class CALL_DETAILS> struct is_call_possible;

namespace detail/*{{{*/
{
	template <class TYPE> class has_member/*{{{*/
	{
		class yes { char m;};
		class no { yes m[2];};

		struct BaseMixin
		{
			void operator()(){}
		};

		struct Base : public TYPE, public BaseMixin {};

		template <class T, T t>  class Helper{};

		// Se o TYPE não tiver um operator(), o U::operator() irá retornar o 
		// BaseMixin::operator(), caso contrário teremos um SFINAE e cairemos
		// no yes deduce(...)
		template <class U>
			static no deduce(U*, Helper<void (BaseMixin::*)(), &U::operator()>* = 0);
		static yes deduce(...);

	public:
		static const bool result = sizeof(yes) == sizeof(deduce((Base*) (0)));

	};/*}}}*/

	template <class TYPE>
		class void_exp_result {};

	template <class TYPE, class U>
		U const& operator,(U const&, void_exp_result<TYPE>);

	template <class TYPE, class U>
		U& operator,(U&, void_exp_result<TYPE>);

	template <class SRC_TYPE, class DEST_TYPE>
		struct clone_constness
		{
			typedef DEST_TYPE type;
		};

	template <class SRC_TYPE, class DEST_TYPE>
		struct clone_constness<const SRC_TYPE, DEST_TYPE>
		{
			typedef const DEST_TYPE type;
		};


	template <class ORIG, class DEST> struct subst_placeholders;/*{{{*/
	template <class R, class DEST, class... ORIG> struct subst_placeholders<R(ORIG...), DEST>
	{
	private:
		template <class T, class enable=void> struct mu
		{
			typedef T type;
		};
		template <class T> struct mu<T,typename std::enable_if<std::is_placeholder<T>::value>::type>
		{
			typedef typename std::tuple_element<std::is_placeholder<T>::value-1,DEST>::type type;
		};
		template <class T> struct mu<std::_Bind<T>>
		{
			typedef typename std::result_of<T>::type type;
		};

	public:
		typedef R type(typename mu<ORIG>::type...);
	};/*}}}*/
	template <class TYPE, class CALL_DETAILS> struct is_bind_call_possible;/*{{{*/
	template<class R, class... ARGS_WANTED, class F, class... ARGS> struct is_bind_call_possible<F(ARGS...), R(ARGS_WANTED...)>

	{
		static const bool value = is_call_possible<F,
					 typename detail::subst_placeholders<R(ARGS...), std::tuple<ARGS_WANTED...>>::type>::value;
	};/*}}}*/

}/*}}}*/

template <class TYPE, class CALL_DETAILS> struct is_call_possible/*{{{*/
{
private:
	class yes {};
	class no { yes m[2]; };

	struct derived : public TYPE
	{
		using TYPE::operator();
		no operator()(...) const;
	};

	typedef typename detail::clone_constness<TYPE, derived>::type
	derived_type;

	template <class T, class DUE_TYPE>
	struct return_value_check
	{
		static yes deduce(DUE_TYPE); // retorno correto
		static no deduce(...); // retorno incorreto
		static no deduce(no); // função não é chamável (c/ relaçãoo aos params)
		// Retorno do TYPE::operator() é 'void'
		static no deduce(detail::void_exp_result<TYPE>);
	};

	template <class T>
	struct return_value_check<T, void>
	{
		static yes deduce(...);
		static no deduce(no);
	};

	template <bool HAS_CALL_OPERATOR, class F>
	struct impl
	{
		// O tipo não tem um operator()
		static const bool value = false;
	};

	template <class T> static T null_object() {}

	template <class R, class... ARGS>
	struct impl<true, R(ARGS...)>
	{
		static const bool value =
			sizeof(
				return_value_check<TYPE, R>::deduce(
					// Se o TYPE::operator() não puder ser chamado com os 
					// parâmetros ARGS, o retorno será 'no'.
					// Caso contrário se o retorno for 'void', 
					// não usaremos os operator,() definidos acima, 
					// e o resultado da expressão é
					// detail::void_exp_result<TYPE>. Se o retorno não for
					// void, então o retorno é o retorno do TYPE::operator()
					(((derived_type*)0)->operator()(null_object<ARGS>()...),
					 detail::void_exp_result<TYPE>())
					)
				) == sizeof(yes);
	};

public:
	static const bool value = impl<detail::has_member<TYPE>::result, CALL_DETAILS>::value;
};/*}}}*/
template <class CALL_DETAILS, class R, class... ARGS> struct is_call_possible<R(ARGS...),CALL_DETAILS>/*{{{*/
{
private:
	struct Functor
	{
		R operator()(ARGS...) {}
	};

public:
	static const bool value = is_call_possible<Functor,CALL_DETAILS>::value;
};/*}}}*/
template <class CALL_DETAILS, class R, class... ARGS> struct is_call_possible<R(*)(ARGS...),CALL_DETAILS>/*{{{*/
{
private:
	struct Functor
	{
		R operator()(ARGS...) {}
	};

public:
	static const bool value = is_call_possible<Functor,CALL_DETAILS>::value;
};/*}}}*/
template <class CALL_DETAILS, class T, class R, class... ARGS> struct is_call_possible<R(T::*)(ARGS...),CALL_DETAILS>/*{{{*/
{
private:
	struct Functor
	{
		R operator()(T*,ARGS...) {}
	};

public:
	static const bool value = is_call_possible<Functor,CALL_DETAILS>::value;
};/*}}}*/
template <class R, class... ARGS_WANTED, class F, class... ARGS> struct is_call_possible<std::_Bind<F(ARGS...)>,R(ARGS_WANTED...)>/*{{{*/
{
	static const bool value 
		= std::conditional<arity<std::_Bind<F(ARGS...)> >::value==sizeof...(ARGS_WANTED),
		detail::is_bind_call_possible<F(ARGS...), R(ARGS_WANTED...)>,
		std::false_type>::type::value;
};/*}}}*/

template <class CALL_DETAILS, class R, class... ARGS> struct is_call_possible<std::function<R(ARGS...)>,CALL_DETAILS>/*{{{*/
{
private:
	struct Functor
	{
		R operator()(ARGS...) {}
	};

public:
	static const bool value = is_call_possible<Functor,CALL_DETAILS>::value;
};/*}}}*/

#ifdef BOOST_FUNCTION_PROLOGUE_HPP
template <class CALL_DETAILS, class R, class... ARGS> struct is_call_possible<boost::function<R(ARGS...)>,CALL_DETAILS>/*{{{*/
{
private:
	struct Functor
	{
		R operator()(ARGS...) {}
	};

public:
	static const bool value = is_call_possible<Functor,CALL_DETAILS>::value;
};/*}}}*/
#endif
#if 0
#ifdef BOOST_BIND_HPP_INCLUDED
template <class R1, class LIST, class R, class... ARGS_WANTED, class F> struct is_call_possible<boost::_bi::bind_t<R1, F, LIST>,R(ARGS_WANTED...)>/*{{{*/
{
	static const bool value 
		= std::conditional<arity<boost::_bi::bind_t<R1,F,LIST>>::value==sizeof...(ARGS_WANTED),
		detail::is_boost_bind_call_possible<F(LIST), R(ARGS_WANTED...)>,
		std::false_type>::type::value;
};/*}}}*/
#endif
#endif

} // namespace s3d

#endif

// $Id: is_call_possible.h 2809 2010-07-01 23:59:39Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

