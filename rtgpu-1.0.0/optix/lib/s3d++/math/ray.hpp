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

template <class T, int D> Ray<T,D> &Ray<T,D>::operator *=(const Matrix<T,D+1,D+1> &m)
{
	origin *= m;
	dir *= m;
	return *this;
}

template <class T, int D> Ray<T,D> &Ray<T,D>::operator +=(const Vector<T,D> &v)
{
	origin += v;
	return *this;
}

template <class T, int D> Ray<T,D> &Ray<T,D>::operator -=(const Vector<T,D> &v)
{
	origin -= v;
	return *this;
}

template <class T, int D> Point<T,D> Ray<T,D>::point_at(real t) const
{
	return origin + t*dir;
}

template <class T, int D> Ray<T,D> unit(const Ray<T,D> &r)
{
	return Ray<T,D>(r.origin, unit(r.dir));
}


}}

// $Id: ray.hpp 2801 2010-06-30 01:57:04Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

