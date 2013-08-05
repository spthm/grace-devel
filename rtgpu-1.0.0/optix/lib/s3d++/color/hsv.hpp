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

#include "pch.h"
#include "../math/r3/linear_transform.h"
#include "cast.h"
#include "hsv.h"
#include "rgb.h"

namespace s3d { namespace color
{


// Ref: http://www.alvyray.com/Papers/hsv2rgb.htm

template <class T>
HSV<T>::HSV(const radiance &r)
{
	using std::max;
	using std::min;

	auto rgb = color_cast<color::rgb>(r);

	auto x = min(rgb.r, min(rgb.g, rgb.b)),
	     v = max(rgb.r, max(rgb.g, rgb.b));

	// Zero saturation?
	if(v == x)
		this->h=0, this->s=0; // Undefined hue, let's set it to zero
	else
	{
		real f;
		if(rgb.r == x)
			f = rgb.g-rgb.b;
		else if(rgb.g == x)
			f = rgb.b-rgb.r;
		else
			f = rgb.r-rgb.g;

		real i;
		if(rgb.r == x)
			i = 3;
		else if(rgb.g == x)
			i = 5;
		else
			i = 1;

		this->h = i-f/(v-x);
		this->s = map<T>((v-x)/v);
	}

	this->v = map<T>(v);
}

template <class T>
HSV<T>::operator radiance() const
{
	using math::mod;
	using std::floor;

	if(s == 0)
		return { unmap(v),unmap(v),unmap(v) };
	else
	{
		real mh = mod(h,6);

		int i = floor(mh);
		real f = mh-i;
		if(!(i&1)) // i is even?
			f = 1-f;

		real s = unmap(this->s),
			 v = unmap(this->v);
			 
		real m = v*(1-s),
			 n = v*(1-s*f);

		switch(i%7)
		{
		case 6:
		case 0:
		default:
			return {v,n,m};
		case 1:
			return {n,v,m};
		case 2:
			return {m,v,n};
		case 3:
			return {m,n,v};
		case 4:
			return {n,m,v};
		case 5:
			return {v,m,n};
		}
	}
}

}} // namespace s3d::color

// $Id: yiq.cpp 2303 2009-06-19 18:04:57Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

