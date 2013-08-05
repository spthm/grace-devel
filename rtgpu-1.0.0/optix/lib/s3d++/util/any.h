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

#ifndef S3D_UTIL_ANY_H
#define S3D_UTIL_ANY_H

#include <stdexcept>
#include "format.h"
#include "demangle.h"
#include "any_fwd.h"

namespace s3d
{

class any
{
	struct safe_bool { int a; };
public:
	any() : m_holder(NULL) {}
	any(const any &that);
	any(any &&that);
	any(any &that); // needed to avoid calling any(T&&v) for T=any&

	template <class T> 
	any(T &&v);

	~any();

	template <class T> 
	any &operator=(T &&v);

	any &operator=(const any &that);
	any &operator=(any &&that);
	any &operator=(any &that); // needed to avoid calling operator=(T&&v) for T=any&

	operator int safe_bool::*() const { return m_holder ? &safe_bool::a : NULL; }
	bool operator!() const { return m_holder ? false : true; }

	const std::type_info &type() const;

	template <class T> bool is_convertible_to() const;

	// Must be defined here to 'try' to avoid ADL doing the wrong thing
	friend std::ostream &operator<<(std::ostream &out, const any &a)
	{
		return out << any_cast<std::string>(a);
	}

private:
	class holder;
	template <class T, class EN=void> class Holder;
	template <class T, class EN=void> class Cast;

	// Should be unique_ptr, but then holder should be defined. We're using
	// raw pointers while gcc doesn't update unique_ptr to c++0x semantics.
	// shared_ptr would be too much...
	holder *m_holder;

	template <class T> friend T any_cast(const any &a);
	template <class T> friend T any_cast(any &&a);
};

class bad_any_cast : public std::bad_cast
{
public:
	bad_any_cast(const std::string &msg) : m_msg(msg) {}
	~bad_any_cast() throw() {}

	virtual const char *what() const throw() { return m_msg.c_str(); }

private:
	std::string m_msg;
};

template <class T, class F>
struct define_type_conversion;


}

#include "any.hpp"

#endif

// $Id: any.h 2859 2010-07-19 00:13:08Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

