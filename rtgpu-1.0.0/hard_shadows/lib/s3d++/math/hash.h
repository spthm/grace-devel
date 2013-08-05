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

#ifndef S3D_MATH_HASH_H
#define S3D_MATH_HASH_H

#include <boost/functional/hash.hpp>

namespace s3d { namespace math
{

template <template<int, class> class A, int N, class T> 
size_t hash_value(const A<N,T> &o)
{
	using namespace boost;
	size_t seed=0;
	for(int i=0; i<N; ++i)
		hash_combine(seed, o[i]);
	return seed;
}

}} // namespace s3d::math

#endif

// $Id: hash.h 2276 2009-06-05 22:49:41Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

