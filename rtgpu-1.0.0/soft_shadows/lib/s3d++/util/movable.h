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

#ifndef S3D_UTIL_MOVABLE_H
#define S3D_UTIL_MOVABLE_H

#include "unique_ptr.h"

namespace s3d
{

class movable
{
public:
    virtual ~movable() {}

    std::unique_ptr<movable> move()
		{ return std::unique_ptr<movable>(do_move()); }
private:
    virtual movable *do_move() = 0;
};

template <class T>
auto to_pointer(T &&obj)
	-> typename std::enable_if<std::is_base_of<movable,T>::value &&
	                           !std::is_const<T>::value,
	    std::unique_ptr<T>>::type
{
	return obj.move();
}

// In the HOLY day when C++ accepts return type covariance of pointer-like
// types (like smart pointers), these macros will be gone for good.

#define DEFINE_MOVABLE(C)\
public: \
    std::unique_ptr<C> move() \
	{ return std::unique_ptr<C>(static_cast<C *>(movable::move().release())); } \
private: \
    virtual C *do_move() { return new C(std::move(*this)); }\

#define DEFINE_PURE_MOVABLE(C)\
public: \
    std::unique_ptr<C> move()  \
	{ return std::unique_ptr<C>(static_cast<C *>(movable::move().release())); } \
private: \
    virtual C *do_move() = 0;

} // namespace s3d

#endif

// $Id: movable.h 2936 2010-08-08 03:31:03Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

