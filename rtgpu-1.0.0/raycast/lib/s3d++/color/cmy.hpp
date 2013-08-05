/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License version 3 as 
	published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public License
	along with S3D++.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "../math/r3/affine_transform.h"
#include "../math/r3/vector.h"
#include "rgb.h"

namespace s3d { namespace color
{

namespace
{

r3::affine_transform rgb_to_cmy { {-1,  0,  0, 0},
							      { 0, -1,  0, 1},
								  { 0,  0, -1, 1} };

}

template <class T>
CMY<T>::CMY(const radiance &v)
{
	using math::map;

	auto aux = rgb_to_cmy * v;
	c = map<T>(aux.x);
	m = map<T>(aux.y);
	y = map<T>(aux.z);
}

template <class T>
CMY<T>::operator radiance() const
{
	// rgb_to_cmy^-1 = rgb_to_cmy
    return rgb_to_cmy * r3::vector(unmap(c),unmap(m),unmap(y)); 
}

}} // namespace s3d::color

// $Id: cmy.hpp 2888 2010-07-27 23:29:05Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

