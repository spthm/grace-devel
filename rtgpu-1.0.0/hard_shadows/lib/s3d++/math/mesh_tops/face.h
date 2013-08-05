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

#ifndef S3D_MATH_FACE_H
#define S3D_MATH_FACE_H

#include "../tops/face.h"

namespace s3d { namespace math
{

namespace detail_math
{
	template <class T>
	struct conv_vertex
	{
		typedef T &result_type;
		T &operator()(tops::vertex &v) const;
	};

	template <class T>
	struct const_conv_vertex
	{
		typedef const T &result_type;
		const T &operator()(const tops::vertex &v) const;
	};
}

template <class T>
class Face : boost::noncopyable, public convertible_from_any

{

public:
	typedef boost::transform_iterator<detail_math::conv_vertex<T>,
									  tops::face_vertex_iterator>
		vertex_iterator;

	typedef boost::transform_iterator<detail_math::const_conv_vertex<T>,
									  tops::const_face_vertex_iterator>
		const_vertex_iterator;

	Face(tops::face &f) : m_face(f) {}

	vertex_iterator vertices_begin();
	vertex_iterator vertices_end();

	T &vertex_at(int i) { return m_face.point_at(i); }
	const T &vertex_at(int i) const { return m_face.point_at(i); }

	const_vertex_iterator vertices_begin() const;
	const_vertex_iterator vertices_end() const;

	tops::face &impl() { return m_face; }
	const tops::face &impl() const { return m_face; }

private:
	tops::face &m_face;
};

}} // namespace s3d::math

#include "face.hpp"

#endif

// $Id: face.h 2725 2010-06-02 19:58:37Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

