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

#include "../tops/face.h"

namespace s3d { namespace math
{

namespace detail_math
{

template <class T>
T &conv_vertex<T>::operator()(tops::vertex &v) const/*{{{*/
{
	return any_cast<T &>(v.user_data());
}/*}}}*/

template <class T>
const T &const_conv_vertex<T>::operator()(const tops::vertex &v) const/*{{{*/
{
	return any_cast<const T &>(v.user_data());
}/*}}}*/

}

template <class T>
typename Face<T>::vertex_iterator Face<T>::vertices_begin()/*{{{*/
{
	return vertex_iterator(m_face.vertices_begin(), detail_math::conv_vertex<T>());
}/*}}}*/

template <class T>
typename Face<T>::vertex_iterator Face<T>::vertices_end()/*{{{*/
{
	return vertex_iterator(m_face.vertices_end(), detail_math::conv_vertex<T>());
}/*}}}*/

template <class T>
typename Face<T>::const_vertex_iterator Face<T>::vertices_begin() const/*{{{*/
{
	return const_vertex_iterator(m_face.vertices_begin(), detail_math::const_conv_vertex<T>());
}/*}}}*/

template <class T>
typename Face<T>::const_vertex_iterator Face<T>::vertices_end() const/*{{{*/
{
	return const_vertex_iterator(m_face.vertices_end(), detail_math::const_conv_vertex<T>());
}/*}}}*/

}} // namespace s3d::math

// $Id: face.hpp 2725 2010-06-02 19:58:37Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

