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

#ifndef S3D_UTIL_CLONABLE_H
#define S3D_UTIL_CLONABLE_H

#include "unique_ptr.h"

namespace s3d
{

class clonable
{
public:
	virtual ~clonable() {}

	std::unique_ptr<clonable> clone() const
	{ return std::unique_ptr<clonable>(do_clone()); }

	friend clonable *new_clone(const clonable &ev)
	{ return ev.do_clone(); }
private:
	virtual clonable *do_clone() const = 0;
};

template <class T> class clonable_ptr
{
	struct dummy { int a; };
public:
	clonable_ptr(T *ptr=NULL) : m_ptr(ptr) {}
	clonable_ptr(std::unique_ptr<T> &&ptr) : m_ptr(std::move(ptr)) {}
	clonable_ptr(clonable_ptr &&that)
		: m_ptr(std::move(that.m_ptr)) {}
	clonable_ptr(const clonable_ptr &that)
		: m_ptr(that.m_ptr?that.m_ptr->clone():std::unique_ptr<T>()) {}

	clonable_ptr &operator=(const clonable_ptr &that)/*{{{*/
	{
		m_ptr = that.m_ptr->clone();
		return *this;
	}/*}}}*/
	clonable_ptr &operator=(clonable_ptr &&that)/*{{{*/
	{
		m_ptr = std::move(that.m_ptr);
		return *this;
	}/*}}}*/

	void reset() { m_ptr.reset(); }
	T *release() { return m_ptr.release(); }

	operator int dummy::*() const/*{{{*/
	{
		if(m_ptr)
			return &dummy::a;
		else
			return NULL;
	}/*}}}*/
	bool operator!() const { return !m_ptr; }

	const T *operator->() const { return m_ptr.get(); }
	T *operator->() { return m_ptr.get(); }

	const T *get() const { return m_ptr.get(); }
	T *get() { return m_ptr.get(); }

private:
	std::unique_ptr<T> m_ptr;
};

template <class T>
auto to_pointer(const T &obj)
	-> typename std::enable_if<std::is_base_of<clonable,T>::value &&
	                           std::is_const<T>::value, 
	    std::unique_ptr<T>>::type
{
	return obj.clone();
}

#define DEFINE_CLONABLE(C)\
public:\
	std::unique_ptr<C> clone() const \
	{ return std::unique_ptr<C>(static_cast<C *>(clonable::clone().release())); } \
private: \
	 virtual C *do_clone() const { return new C(*this); }\

#define DEFINE_PURE_CLONABLE(C)\
public:\
	std::unique_ptr<C> clone() const \
	{ return std::unique_ptr<C>(static_cast<C *>(clonable::clone().release())); } \
private: \
	 virtual C *do_clone() const = 0;

} // namespace s3d

#endif

// $Id: clonable.h 2936 2010-08-08 03:31:03Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

