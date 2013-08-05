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

#ifndef S3D_UTIL_HASH_H
#define S3D_UTIL_HASH_H

#include <boost/functional/hash.hpp>

namespace std
{
template<class T> size_t hash<T>::operator()(T v) const
{
    return hash_value(v);
}
} // namespace s3d

namespace s3d
{
	inline size_t hash_combine(size_t &seed)
	{
		return seed;
	}

	template <class T, class... ARGS> 
	size_t hash_combine(size_t &seed, const T &a1, const ARGS &...args)
	{
		boost::hash_combine(seed, a1);
		hash_combine(seed, hash_combine(seed, args...));
		return seed;
	}

	template <class... ARGS> 
	size_t hash_combine(const ARGS &...args)
	{
		size_t seed = 0;
		return hash_combine(seed, args...);
	}
}

#endif

// $Id: hash.h 2887 2010-07-27 13:41:39Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

