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

#include "../tops/surface.h"

namespace s3d { namespace math
{

namespace detail_math
{

template <class T>
Face<T> &conv_face<T>::operator()(tops::face &f) const/*{{{*/
{
	if(!f.user_data())
		f.set_user_data(make_unique<Face<T>>(f));

	return any_cast<Face<T> &>(f.user_data());
}/*}}}*/

template <class T>
const Face<T> &const_conv_face<T>::operator()(const tops::face &f) const/*{{{*/
{
	return conv_face<T>()(const_cast<tops::face &>(f));
}/*}}}*/

}

template <class T>
typename Surface<T>::face_iterator Surface<T>::faces_begin()/*{{{*/
{
	return face_iterator(m_surface.faces_begin(),detail_math::conv_face<T>());
}/*}}}*/

template <class T>
typename Surface<T>::face_iterator Surface<T>::faces_end()/*{{{*/
{
	return face_iterator(m_surface.faces_end(),detail_math::conv_face<T>());
}/*}}}*/

template <class T>
typename Surface<T>::const_face_iterator Surface<T>::faces_begin() const/*{{{*/
{
	return const_face_iterator(m_surface.faces_begin(),detail_math::const_conv_face<T>());
}/*}}}*/

template <class T>
typename Surface<T>::const_face_iterator Surface<T>::faces_end() const/*{{{*/
{
	return const_face_iterator(m_surface.faces_end(),detail_math::const_conv_face<T>());
}/*}}}*/

template <class T>
Face<T> *Surface<T>::create_face(const T &v1, const T &v2, const T &v3)/*{{{*/
{
	return &detail_math::conv_face<T>()(*m_surface.create_face(v1,v2,v3));
}/*}}}*/

template <class T> 
bool Surface<T>::erase_face(Face<T> *f)/*{{{*/
{
	if(f)
		return m_surface.erase_face(&f->impl());
	else
		return false;
}/*}}}*/

}} // namespace s3d::math

// $Id: surface.hpp 2725 2010-06-02 19:58:37Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

