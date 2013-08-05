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

#ifndef S3D_MESH_MESH_H
#define S3D_MESH_MESH_H

#include <vector>
#include "../util/flat_view.h"
#include "surface.h"

namespace s3d { namespace math
{

template <class M>
struct is_mesh
{
	static const bool value = false;
};

template <class M>
struct is_mesh<M&> : is_mesh<M>
{
};

template <class M>
struct is_mesh<const M> : is_mesh<M>
{
};

template <class S>
class mesh
{
public:
	typedef typename std::conditional<is_surface<S>::value,S,surface<S>>::type 
		surface_type;
private:
	typedef std::vector<surface_type> surfaces;
public:
	typedef typename surface_type::value_type face_type;
	typedef typename face_type::value_type vertex_type;

	typedef surface_type value_type;

	typedef typename surfaces::iterator iterator;
	typedef typename surfaces::const_iterator const_iterator;

	mesh() {}

	mesh(const surface_type &surf) { m_surfaces.push_back(surf); }
	mesh(surface_type &&surf) { m_surfaces.push_back(std::move(surf)); }

	mesh(const std::vector<face_type> &faces);
	mesh(std::vector<face_type> &&faces);

	mesh(const mesh &that) = default;
	mesh(mesh &&that) : m_surfaces(std::move(that.m_surfaces)) {}

	template <class U, class = typename std::enable_if<
		std::is_convertible<typename mesh<U>::vertex_type, vertex_type>::value>::type>
	explicit mesh(const mesh<U> &that);

	template <class U, class = typename std::enable_if<
		std::is_convertible<typename mesh<U>::vertex_type, vertex_type>::value>::type>
	explicit mesh(mesh<U> &&that);

	mesh &operator=(const mesh &that) = default;
	mesh &operator=(mesh &&that) 
		{ m_surfaces = std::move(that.m_surfaces); return *this; }

	bool operator==(const mesh &that) const
		{ return m_surfaces == that.m_surfaces; }
	bool operator!=(const mesh &that) const
		{ return !operator==(that); }

	auto faces() -> flat_view<mesh,1> 
		{ return *this | flattened<1>(); }
	auto faces() const -> flat_view<const mesh,1> 
		{ return *this | flattened<1>(); }

	auto vertices() -> flat_view<mesh,2> 
		{ return *this | flattened<2>(); }
	auto vertices() const -> flat_view<const mesh,2> 
		{ return *this | flattened<2>(); }

	auto begin() -> iterator { return m_surfaces.begin(); }
	auto end() -> iterator { return m_surfaces.end(); }

	auto begin() const -> const_iterator { return m_surfaces.begin(); }
	auto end() const -> const_iterator { return m_surfaces.end(); }

	void reserve(unsigned s) { m_surfaces.reserve(s); }
	size_t size() const { return m_surfaces.size(); }
	bool empty() const { return m_surfaces.empty(); }

#if 0
	template <class...ARGS>
	auto make_surface(ARGS &&...args) -> surface_type *;
#endif
	void push_back(const surface_type &surf) 
		{ m_surfaces.push_back(surf); }
	void push_back(surface_type &&surf) 
		{ m_surfaces.push_back(std::move(surf)); }

	auto make_surface(const std::initializer_list<face_type> &faces)
		-> surface_type *;

	bool erase_surface(surface_type *s);

private:
	surfaces m_surfaces;
};

template <class S>
struct is_mesh<mesh<S>>
{
	static const bool value = true;
};

template <class T>
class mesh_view : mesh<surface_view<T>>
{
public:
	template <class...ARGS>
	mesh_view(ARGS &&...args) 
		: mesh<surface_view<T>>(std::forward<ARGS>(args)...)
	{
	}
};

template <class T>
struct is_mesh<mesh_view<T>>
{
	static const bool value = true;
};

template <class S>
std::ostream &operator<<(std::ostream &out, const mesh<S> &mesh);

}} // namespace s3d::math

#include "mesh.hpp"

#endif

// $Id: mesh.h 3112 2010-09-06 01:30:30Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

