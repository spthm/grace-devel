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

template <class S> 
mesh<S>::mesh(const std::vector<face_type> &faces) 
{ 
	m_surfaces.push_back(surface_type(faces));
}

template <class S> 
mesh<S>::mesh(std::vector<face_type> &&faces) 
{ 
	m_surfaces.push_back(surface_type(std::move(faces)));
}

template <class S> template <class U, class>
mesh<S>::mesh(const mesh<U> &that)
{
	m_surfaces.reserve(that.size());

	BOOST_FOREACH(auto &surf, that)
		m_surfaces.emplace_back(surf);
}

template <class S> template <class U, class>
mesh<S>::mesh(mesh<U> &&that)
{
	m_surfaces.reserve(that.size());

	BOOST_FOREACH(auto &surf, that)
		m_surfaces.emplace_back(std::move(surf));
}

#if 0
template <class S> template <class...ARGS>
auto mesh<S>::make_surface(ARGS &&...args) -> surface_type * /*{{{*/
{
	m_surfaces.emplace_back(std::forward<ARGS>(args)...);
	return &m_surfaces.back();
}/*}}}*/
#endif

template <class S> 
auto mesh<S>::make_surface(const std::initializer_list<face_type> &faces) 
	-> surface_type *
{
	m_surfaces.push_back(surface_type(faces));
	return &m_surfaces.back();
}

template <class S> 
bool mesh<S>::erase_surface(surface_type *s)/*{{{*/
{
	// Just for completeness, should'nt be called frequently as it is O(n)
	for(auto it=m_surfaces.begin(); it!=m_surfaces.end(); ++it)
	{
		if(&*it == s)
		{
			m_surfaces.erase(it);
			return true;
		}
	}
	return false;
}/*}}}*/

template <class S>
std::ostream &operator<<(std::ostream &out, const mesh<S> &mesh)/*{{{*/
{
	out << '{';

	if(!mesh.empty())
	{
		auto itv = mesh.begin();

		out << *itv++;

		for(;itv != mesh.end(); ++itv)
			out << ',' << *itv;
	}

	return out << '}';
}/*}}}*/

}} // namespace s3d::math

// $Id: mesh.hpp 3112 2010-09-06 01:30:30Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

