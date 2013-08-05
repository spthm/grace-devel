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

#include "../util/function_traits.h"

namespace s3d { namespace math
{

namespace detail_math
{

template <class T>
Surface<T> &conv_surface<T>::operator()(tops::surface &s) const/*{{{*/
{
	if(!s.user_data())
		s.set_user_data(make_unique<Surface<T>>(s));

	return any_cast<Surface<T>&>(s.user_data());
}/*}}}*/

template <class T>
const Surface<T> &const_conv_surface<T>::operator()(const tops::surface &s) const/*{{{*/
{
	return conv_surface<T>()(const_cast<tops::surface &>(s));
}/*}}}*/

}

template <class T> 
Mesh<T>::Mesh(const Mesh &that)/*{{{*/
	: m_mesh(that.m_mesh)
{
}/*}}}*/

template <class T> 
Mesh<T>::Mesh(Mesh &&that)/*{{{*/
	: m_mesh(std::move(that.m_mesh))
{
}/*}}}*/

template <class T> 
Mesh<T> &Mesh<T>::operator=(const Mesh &that)/*{{{*/
{
	m_mesh = that.m_mesh;
	return *this;
}/*}}}*/

template <class T> 
Mesh<T> &Mesh<T>::operator=(Mesh &&that)/*{{{*/
{
	m_mesh = std::move(that.m_mesh);
	return *this;
}/*}}}*/

template <class T>
typename Mesh<T>::surface_iterator Mesh<T>::surfaces_begin()/*{{{*/
{
	return surface_iterator(m_mesh.surfaces_begin(),detail_math::conv_surface<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::surface_iterator Mesh<T>::surfaces_end()/*{{{*/
{
	return surface_iterator(m_mesh.surfaces_end(),detail_math::conv_surface<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::const_surface_iterator Mesh<T>::surfaces_begin() const/*{{{*/
{
	return const_surface_iterator(m_mesh.surfaces_begin(),detail_math::const_conv_surface<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::const_surface_iterator Mesh<T>::surfaces_end() const/*{{{*/
{
	return const_surface_iterator(m_mesh.surfaces_end(),detail_math::const_conv_surface<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::face_iterator Mesh<T>::faces_begin()/*{{{*/
{
	return face_iterator(m_mesh.faces_begin(),detail_math::conv_face<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::face_iterator Mesh<T>::faces_end()/*{{{*/
{
	return face_iterator(m_mesh.faces_end(),detail_math::conv_face<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::const_face_iterator Mesh<T>::faces_begin() const/*{{{*/
{
	return const_face_iterator(m_mesh.faces_begin(),detail_math::const_conv_face<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::const_face_iterator Mesh<T>::faces_end() const/*{{{*/
{
	return const_face_iterator(m_mesh.faces_end(),detail_math::const_conv_face<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::vertex_iterator Mesh<T>::vertices_begin()/*{{{*/
{
	return vertex_iterator(m_mesh.vertices_begin(),detail_math::conv_vertex<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::vertex_iterator Mesh<T>::vertices_end()/*{{{*/
{
	return vertex_iterator(m_mesh.vertices_end(),detail_math::conv_vertex<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::const_vertex_iterator Mesh<T>::vertices_begin() const/*{{{*/
{
	return const_vertex_iterator(m_mesh.vertices_begin(),detail_math::const_conv_vertex<T>());
}/*}}}*/

template <class T>
typename Mesh<T>::const_vertex_iterator Mesh<T>::vertices_end() const/*{{{*/
{
	return const_vertex_iterator(m_mesh.vertices_end(),detail_math::const_conv_vertex<T>());
}/*}}}*/

template <class T>
Surface<T> *Mesh<T>::create_surface()/*{{{*/
{
	return &detail_math::conv_surface<T>()(*m_mesh.create_surface());
}/*}}}*/

template <class T> 
bool Mesh<T>::erase_surface(Surface<T> *s)/*{{{*/
{
	if(s)
		return m_mesh.erase_surface(&s->impl());
	else
		return false;
}/*}}}*/

template <class T, class F, class C>
Mesh<T> &mesh_cast(Mesh<F> &&m, C conv)/*{{{*/
{
	static_assert(
		std::is_convertible<typename std::result_of<C(F())>::type, T>::value,
		   "Cannot convert mesh to desired vertex type"); 

	for(tops::mesh_vertex_iterator it = m.impl().vertices_begin();
		it!=m.impl().vertices_end(); ++it)
	{
		it->set_user_data(conv(any_cast<F&>(it->user_data())));
	}

	return *reinterpret_cast<Mesh<T> *>(&m);
}/*}}}*/

template <class T, class F, class C>
std::unique_ptr<Mesh<T>> mesh_cast(std::unique_ptr<Mesh<F>> m, C conv)/*{{{*/
{
	if(m)
	{
		mesh_cast<T>(*m, conv);

		// Ol' dirty type-punning based trick to avoid breaking strict aliasing
		// rules
		union
		{
			std::unique_ptr<Mesh<F>> *in;
			std::unique_ptr<Mesh<T>> *out;
		} cast;
		cast.in = &m;

		return std::move(*cast.out);
	}
	else
		return std::unique_ptr<Mesh<T>>();
}/*}}}*/

}} // namespace s3d::math

// $Id: mesh.hpp 2725 2010-06-02 19:58:37Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

