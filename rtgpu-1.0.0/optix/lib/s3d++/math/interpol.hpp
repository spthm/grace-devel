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

#include "../util/type_traits.h"

namespace s3d { namespace math
{

template <class T> 
T lerp(real t, const T &a, const T &b)/*{{{*/
{
	return a + (b-a)*t;
}/*}}}*/
template <class T> 
T bilerp(real t, real ta, const T &a0, const T &a1,/*{{{*/
				 real tb, const T &b0, const T &b1)
{
	return lerp(t, lerp(ta, a0, a1), lerp(tb, b0, b1));
}/*}}}*/

template <class T> 
T bezier(real t, const T &a, const T &b)/*{{{*/
{
	return lerp(t, a, b);
}/*}}}*/
template <class T> 
T bezier(real t, const T &a, const T &b, const T &c)/*{{{*/
{
	T ab = lerp(t, a, b),
	  bc = lerp(t, b, c);
	return bezier(t, ab, bc);
}/*}}}*/
template <class T> 
T bezier(real t, const T &a, const T &b, const T &c, const T &d)/*{{{*/
{
	T ab = lerp(t, a, b),
	  bc = lerp(t, b, c),
	  cd = lerp(t, c, d);
	return bezier(t, ab, bc, cd);
}/*}}}*/

}} // s3d::math


// $Id: interpol.hpp 2944 2010-08-09 03:59:27Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

