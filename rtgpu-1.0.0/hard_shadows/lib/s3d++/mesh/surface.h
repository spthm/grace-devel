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

#ifndef S3D_MESH_SURFACE_H
#define S3D_MESH_SURFACE_H

#include <vector>
#include "face.h"
#include "../util/flat_view.h"

namespace s3d { namespace math
{

template <class S>
struct is_surface
{
	static const bool value = false;
};

template <class M>
struct is_surface<M&> : is_surface<M>
{
};

template <class M>
struct is_surface<const M> : is_surface<M>
{
};

template <class F>
class surface 
{
public:
	typedef typename std::conditional<is_face<F>::value, F, face<F>>::type 
		face_type;

private:
	typedef std::vector<face_type> faces;
public:
	typedef typename faces::iterator iterator;
	typedef typename faces::const_iterator const_iterator;

	typedef typename face_type::value_type vertex_type;

	typedef face_type value_type;

	surface() {}
	surface(const std::vector<face_type> &faces) : m_faces(faces) {}
	surface(std::vector<face_type> &&faces) : m_faces(std::move(faces)) {}

	template <class... V>
	surface(const face_type &v0, V &&...faces);

	template <class... V>
	surface(face_type &&v0, V &&...faces);

	surface(const std::initializer_list<face_type> &faces);

	surface(const surface &that);
	surface(surface &&that);

	template <class U, class = typename std::enable_if<
	   std::is_convertible<typename surface<U>::vertex_type,
						   vertex_type>::value>::type>
	explicit surface(const surface<U> &that);

	template <class U, class = typename std::enable_if<
	   std::is_convertible<typename surface<U>::vertex_type,
						   vertex_type>::value>::type>
	explicit surface(surface<U> &&that);

	bool operator==(const surface &that) const
		{ return m_faces == that.m_faces; }
	bool operator!=(const surface &that) const
		{ return !operator==(that); }

	surface &operator=(const surface &that) = default;
	surface &operator=(surface &&that) 
		{ m_faces = std::move(that.m_faces); return *this; }

	face_type &back() 
		{ assert(!empty()); return *m_faces.rbegin(); }
	const face_type &back() const 
		{ assert(!empty()); return *m_faces.rbegin(); }

	face_type &front() 
		{ assert(!empty()); return *m_faces.begin(); }
	const face_type &front() const 
		{ assert(!empty()); return *m_faces.begin(); }

	iterator begin() { return m_faces.begin(); }
	iterator end() { return m_faces.end(); }

	const_iterator begin() const { return m_faces.begin(); }
	const_iterator end() const { return m_faces.end(); }

	auto vertices() -> flat_view<surface,1> 
		{ return *this | flattened<1>(); }
	auto vertices() const -> flat_view<const surface,1> 
		{ return *this | flattened<1>(); }

	size_t size() const { return m_faces.size(); }
	bool empty() const { return m_faces.empty(); }

	void face_reserve(unsigned s) { m_faces.reserve(s); }
	void reserve(unsigned s) { m_faces.reserve(s); }

	void push_back(const face_type &face) { m_faces.push_back(face); }
	void push_back(face_type &&face) { m_faces.push_back(std::move(face)); }

	template <class...V>
	face_type *make_face(V &&...verts);

	bool erase_face(face_type *s);

private:
	faces m_faces;
};

template <class F>
struct is_surface<surface<F>>
{
	static const bool value = true;
};

template <class T>
class surface_view : surface<face_view<T>>
{
public:
	template <class...ARGS>
	surface_view(ARGS &&...args) 
		: surface<face_view<T>>(std::forward<ARGS>(args)...)
	{
	}
};

template <class F>
struct is_surface<surface_view<F>>
{
	static const bool value = true;
};


template <class F>
std::ostream &operator<<(std::ostream &out, const surface<F> &surface);


auto make_triangle_grid(std::vector<r2::param_coord> &coords,
						  const r2::box &a, size_t nu, size_t nv)
	-> surface<face<int,3>>;

auto make_triangle_strip_grid(std::vector<r2::param_coord> &coords,
							    const r2::box &a, size_t nu, size_t nv)
	-> surface<triangle_strip<int>>;

}} // namespace s3d::math

#include "surface.hpp"

#endif

// $Id: surface.h 3112 2010-09-06 01:30:30Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

