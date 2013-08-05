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

#ifndef S3D_UTIL_CLASS_POOL_H
#define S3D_UTIL_CLASS_POOL_H

#include <boost/pool/object_pool.hpp>
#include <iostream>
#include <cassert>

namespace s3d
{

template <class T>
class class_memory_pool
{
public:
	void *operator new(size_t s, void *p) // placement new
	{
		return ::operator new(s,p);
	}

	void *operator new(size_t s)
	{
		assert(s <= sizeof(T));
		return m_pool.malloc();
	}

	void *operator new[](size_t s)
	{
		return m_pool.ordered_malloc(s/sizeof(T)+1);
	}

	void *operator new(size_t s, std::nothrow_t nt) throw()
	{
		try
		{
			return operator new(s);
		}
		catch(std::bad_alloc &)
		{
			return NULL;
		}
	}

	void operator delete(void *p, size_t s)
	{
		m_pool.free(p);
	}

	void operator delete[](void *p, size_t s)
	{
		m_pool.ordered_free(p, s/sizeof(T)+1);
	}

private:
	static boost::pool<> m_pool;
};

template <class T> boost::pool<> class_memory_pool<T>::m_pool(sizeof(T),512);

}

#endif

// $Id: class_pool.h 2298 2009-06-09 15:24:25Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

