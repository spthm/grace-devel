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

#ifndef S3D_UTIL_CREATABLE_H
#define S3D_UTIL_CREATABLE_H

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/seq.hpp>
#include <boost/preprocessor/seq/to_tuple.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/tuple/rem.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/control/if.hpp>
#include "unique_ptr.h"

namespace s3d
{

template <class...ARGS>
class creatable
{
public:
    virtual ~creatable() {}

    std::unique_ptr<creatable> create(ARGS... args) const
		{ return std::unique_ptr<creatable>(do_create(std::move(args)...)); }
private:
    virtual creatable *do_create(ARGS... args) const = 0;
};

// This is ugly beyond repair.

#define S3D_VA_NUM_ARGS(...) \
	S3D_VA_NUM_ARGS_IMPL(__VA_ARGS__,10,9,8,7,6,5,4,3,2,1)
#define S3D_VA_NUM_ARGS_IMPL(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,N,...) N

#ifndef BOOST_PP_SEQ_ENUM_0
#	define BOOST_PP_SEQ_ENUM_0
#endif

#define S3D_ARGS_TO_SEQ(...) \
		BOOST_PP_TUPLE_TO_SEQ(S3D_VA_NUM_ARGS(__VA_ARGS__), (__VA_ARGS__))

#define S3D_DECL_ARG(r, _, i, elem) (elem BOOST_PP_CAT(arg,i))
#define S3D_MOVE_ARG(r, _, i, elem) (std::move(BOOST_PP_CAT(arg,i)))

#define S3D_DECL_ARGS(...) \
	BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_FOR_EACH_I(S3D_DECL_ARG,_,\
						BOOST_PP_SEQ_TAIL(S3D_ARGS_TO_SEQ(__VA_ARGS__))))

#define S3D_MOVE_ARGS(...) \
	BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_FOR_EACH_I(S3D_MOVE_ARG,_,\
						BOOST_PP_SEQ_TAIL(S3D_ARGS_TO_SEQ(__VA_ARGS__))))

#define S3D_ARG_HEAD(...) BOOST_PP_SEQ_HEAD(S3D_ARGS_TO_SEQ(__VA_ARGS__))
#define S3D_ARG_TAIL(...) \
	BOOST_PP_IF(BOOST_PP_DEC(S3D_VA_NUM_ARGS(__VA_ARGS__)), \
				BOOST_PP_SEQ_ENUM, \
				BOOST_PP_TUPLE_REM(1))\
		(BOOST_PP_SEQ_TAIL(S3D_ARGS_TO_SEQ(__VA_ARGS__)))

#define DEFINE_CREATABLE(...) \
public: \
    std::unique_ptr<S3D_ARG_HEAD(__VA_ARGS__)> create(S3D_DECL_ARGS(__VA_ARGS__)) const\
	{ return std::unique_ptr<S3D_ARG_HEAD(__VA_ARGS__)>(static_cast<S3D_ARG_HEAD(__VA_ARGS__) *>(creatable<S3D_ARG_TAIL(__VA_ARGS__)>::create(S3D_MOVE_ARGS(__VA_ARGS__)).release())); } \
private: \
    virtual S3D_ARG_HEAD(__VA_ARGS__) *do_create(S3D_DECL_ARGS(__VA_ARGS__)) const \
		{ return new S3D_ARG_HEAD(__VA_ARGS__)(S3D_MOVE_ARGS(__VA_ARGS__)); }\

#define DEFINE_PURE_CREATABLE(...)\
public: \
    std::unique_ptr<S3D_ARG_HEAD(__VA_ARGS__)> create(S3D_DECL_ARGS(__VA_ARGS__))  const \
	{ return std::unique_ptr<S3D_ARG_HEAD(__VA_ARGS__)>(static_cast<S3D_ARG_HEAD(__VA_ARGS__) *>(creatable<S3D_ARG_TAIL(__VA_ARGS__)>::create(S3D_MOVE_ARGS(__VA_ARGS__)).release())); } \
private: \
    virtual S3D_ARG_HEAD(__VA_ARGS__) *do_create(S3D_DECL_ARGS(__VA_ARGS__)) const = 0;

} // namespace s3d

#endif

// $Id: creatable.h 2726 2010-06-02 19:58:56Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

