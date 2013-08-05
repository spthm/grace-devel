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

namespace s3d { namespace math
{

template <class F>
surface<F>::surface(const surface &that)/*{{{*/
	: m_faces(that.m_faces)
{
}/*}}}*/

template <class F>
surface<F>::surface(surface &&that)/*{{{*/
	: m_faces(std::move(that.m_faces))
{
}/*}}}*/

template <class F> template <class... FACES>
surface<F>::surface(const face_type &face, FACES &&...faces)/*{{{*/
	: m_faces(surface::faces{{face, std::forward<FACES>(faces)...}})
{
}/*}}}*/

template <class F> template <class... FACES>
surface<F>::surface(face_type &&face, FACES &&...faces)/*{{{*/
	: m_faces(surface::faces{{std::move(face), std::forward<FACES>(faces)...}})
{
}/*}}}*/

template <class F>
surface<F>::surface(const std::initializer_list<face_type> &faces)/*{{{*/
	: m_faces(faces.begin(), faces.end())
{
}/*}}}*/

template <class S> template <class U, class>
surface<S>::surface(const surface<U> &that)/*{{{*/
{
	m_faces.reserve(that.size());

	BOOST_FOREACH(auto &face, that)
		m_faces.emplace_back(face);
}/*}}}*/

template <class S> template <class U, class>
surface<S>::surface(surface<U> &&that)/*{{{*/
{
	m_faces.reserve(that.size());

	BOOST_FOREACH(auto &face, that)
		m_faces.emplace_back(std::move(face));
}/*}}}*/

template <class F> template <class... V>
auto surface<F>::make_face(V &&...verts) -> face_type * /*{{{*/
{
	m_faces.emplace_back(std::forward<V>(verts)...);
	return &m_faces.back();
}/*}}}*/

template <class F> 
bool surface<F>::erase_face(face_type *f)/*{{{*/
{
	// Just for completion, shouldn't be called frequently as it's costly
	for(auto it=m_faces.begin(); it!=m_faces.end(); ++it)
	{
		if(&*it == f)
		{
			m_faces.erase(it);
			return true;
		}
	}
	return false;
}/*}}}*/

template <class F>
std::ostream &operator<<(std::ostream &out, const surface<F> &surface)/*{{{*/
{
	out << '{';

	if(!surface.empty())
	{
		auto itv = surface.begin();

		out << *itv++;

		for(;itv != surface.end(); ++itv)
			out << ',' << *itv;
	}

	return out << '}';
}/*}}}*/

}} // namespace s3d::math

// $Id: surface.hpp 3112 2010-09-06 01:30:30Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

