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

#ifndef S3D_MATH_MESH_H
#define S3D_MATH_MESH_H

#include <boost/iterator/transform_iterator.hpp>
#include "../tops/mesh.h"
#include "surface.h"

namespace s3d { namespace math
{

namespace detail_math
{
	template <class T>
	struct conv_surface
	{
		typedef Surface<T> &result_type;
		Surface<T> &operator()(tops::surface &s) const;
	};
	template <class T>
	struct const_conv_surface
	{
		typedef const Surface<T> &result_type;
		const Surface<T> &operator()(const tops::surface &s) const;
	};
}

template <class T>
class Mesh
{
public:
	typedef boost::transform_iterator<detail_math::conv_surface<T>,
							  tops::surface_iterator> 
		surface_iterator;

	typedef boost::transform_iterator<detail_math::const_conv_surface<T>,
							  tops::const_surface_iterator>
		const_surface_iterator;

	typedef boost::transform_iterator<detail_math::conv_face<T>,
							  tops::mesh_face_iterator> 
		face_iterator;

	typedef boost::transform_iterator<detail_math::const_conv_face<T>,
							  tops::const_mesh_face_iterator>
		const_face_iterator;

	typedef boost::transform_iterator<detail_math::conv_vertex<T>,
							  tops::mesh_vertex_iterator> 
		vertex_iterator;

	typedef boost::transform_iterator<detail_math::const_conv_vertex<T>,
							  tops::const_mesh_vertex_iterator>
		const_vertex_iterator;

	Mesh() {}
	Mesh(const Mesh &that);
	Mesh(Mesh &&that);

	Mesh &operator=(const Mesh &that);
	Mesh &operator=(Mesh &&that);

	surface_iterator surfaces_begin();
	surface_iterator surfaces_end();

	const_surface_iterator surfaces_begin() const;
	const_surface_iterator surfaces_end() const;

	face_iterator faces_begin();
	face_iterator faces_end();

	const_face_iterator faces_begin() const;
	const_face_iterator faces_end() const;

	vertex_iterator vertices_begin();
	vertex_iterator vertices_end();

	const_vertex_iterator vertices_begin() const;
	const_vertex_iterator vertices_end() const;

	Surface<T> *create_surface();

	bool erase_surface(Surface<T> *s);

	tops::mesh &impl() { return m_mesh; }
	const tops::mesh &impl() const { return m_mesh; }

private:
	tops::mesh m_mesh;
};

template <class T, class F, class C>
std::unique_ptr<Mesh<T>> mesh_cast(std::unique_ptr<Mesh<F>> m, C conv);

template <class T, class F, class C>
Mesh<T> &mesh_cast(Mesh<F> &&m, C conv);


}} // namespace s3d::math

#include "mesh.hpp"

#endif

// $Id: mesh.h 2725 2010-06-02 19:58:37Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

