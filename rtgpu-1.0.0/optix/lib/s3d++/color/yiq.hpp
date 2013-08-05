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

#include "../math/r3/linear_transform.h"
#include "rgb.h"

namespace s3d { namespace color
{

namespace
{

r3::linear_transform rgb_to_yiq{{0.2989,  0.5866,  0.1144},
							    {0.5959, -0.2741, -0.3218},
								{0.2113, -0.5227,  0.3113}};
}

template <class T>
YIQ<T>::YIQ(const radiance &v)
{
	auto aux = rgb_to_yiq * v;
	y = map<T>(aux.x);
	i = map<T>(aux.y);
	q = map<T>(aux.z);
}

template <class T>
YIQ<T>::operator radiance() const
{
    return inv(rgb_to_yiq) * radiance(unmap(y),unmap(i),unmap(q));
}

}} // namespace s3d::color

// $Id: yiq.hpp 2874 2010-07-21 13:33:16Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

