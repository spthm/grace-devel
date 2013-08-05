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

#ifndef S3D_MATH_SURFACE_H
#define S3D_MATH_SURFACE_H

#include "../tops/surface.h"
#include "face.h"

namespace s3d { namespace math
{

namespace detail_math
{
	template <class T>
	struct conv_face
	{
		typedef Face<T> &result_type;
		Face<T> &operator()(tops::face &s) const;
	};

	template <class T>
	struct const_conv_face
	{
		typedef const Face<T> &result_type;
		const Face<T> &operator()(const tops::face &s) const;
	};
}

template <class T>
class Surface : boost::noncopyable, public convertible_from_any
{
public:
	typedef boost::transform_iterator<detail_math::conv_face<T>,
								  tops::face_iterator> 
		face_iterator;

	typedef boost::transform_iterator<detail_math::const_conv_face<T>,
								  tops::const_face_iterator>
		const_face_iterator;

	Surface(tops::surface &s) : m_surface(s) {}

	face_iterator faces_begin();
	face_iterator faces_end();

	const_face_iterator faces_begin() const;
	const_face_iterator faces_end() const;

	Face<T> *create_face(const T &p1, const T &p2, const T &p3);

	bool erase_face(Face<T> *s);

	tops::surface &impl() { return m_surface; }
	const tops::surface &impl() const { return m_surface; }

private:
	tops::surface &m_surface;
};

}} // namespace s3d::math

#include "surface.hpp"

#endif

// $Id: surface.h 2725 2010-06-02 19:58:37Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

