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

#include <cassert>
#include <boost/foreach.hpp>
#include "../util/type_traits.h"
#include "../math/point.h"

namespace s3d { namespace math
{

template <class T>
triangle_adjacency_strip<T>::triangle_adjacency_strip(const std::initializer_list<T> &vtx) /*{{{*/
	: m_vertices(vtx.begin(), vtx.end())
{
}/*}}}*/

template <class T> template <class IT, class IT2>
void triangle_adjacency_strip<T>::insert(IT itpos, IT2 itbeg, IT2 itend)/*{{{*/
{
	m_vertices.insert(itpos, itbeg, itend);
}/*}}}*/

template <class T> template <class...ARGS>
void triangle_adjacency_strip<T>::push_back(ARGS &&...args) /*{{{*/
{ 
	m_vertices.push_back(std::forward<ARGS>(args)...);
}/*}}}*/

template <class T>
T &triangle_adjacency_strip<T>::operator[](size_t i) /*{{{*/
{ 
	assert(i < m_vertices.size());
	return m_vertices[i];
}/*}}}*/
template <class T>
const T &triangle_adjacency_strip<T>::operator[](size_t i) const /*{{{*/
{ 
	assert(i < m_vertices.size());
	return m_vertices[i];
}/*}}}*/

template <class T>
T &triangle_adjacency<T>::operator[](size_t i) /*{{{*/
{ 
	assert(i < m_vertices.size());
	return m_vertices[i];
}/*}}}*/
template <class T>
const T &triangle_adjacency<T>::operator[](size_t i) const /*{{{*/
{ 
	assert(i < m_vertices.size());
	return m_vertices[i];
}/*}}}*/

}} // namespace s3d::math

// $Id: triangle_adjacency_strip.hpp 3110 2010-09-05 17:25:29Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

