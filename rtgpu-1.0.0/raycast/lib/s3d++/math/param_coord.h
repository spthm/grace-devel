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

#ifndef S3D_MATH_PARAM_COORD_H
#define S3D_MATH_PARAM_COORD_H

#include <boost/operators.hpp>
#include "real.h"
#include "fwd.h"
#include "coords.h"
#include "vector.h"
#include "size.h"
#include "../mpl/vector_fwd.h"
#include "param_space.h"

namespace s3d { namespace math
{

template<class T, int D> 
struct ParamCoord 
	: boost::totally_ordered<ParamCoord<T,D>> 
	, coords<ParamCoord<T,D>,param_space<T,D>>
{
private:
	typedef coords<ParamCoord,param_space<T,D>> coords_base;
public:
	template <class U, int N=D> struct rebind {typedef ParamCoord<U,N> type;};

	ParamCoord() {}
	ParamCoord(const T &v);
	template <class U> ParamCoord(const Point<U,D> &p);

	template <class... ARGS> 
	ParamCoord(T c1, ARGS... cn) : coords_base(c1, cn...) {}

	template <class U> ParamCoord(const ParamCoord<U,D> &that);
	ParamCoord(const ParamCoord &that);

	operator Point<T,D>() const;

	ParamCoord &operator+=(const Vector<T,D> &v);
	ParamCoord &operator-=(const Vector<T,D> &v);

	ParamCoord &operator*=(const Size<T,D> &s);
	ParamCoord &operator/=(const Size<T,D> &s);

	Vector<T,D> operator-(const ParamCoord &that) const;

	bool operator==(const ParamCoord &that) const;

	ParamCoord &operator +=(T d);
	ParamCoord &operator -=(T d);
	ParamCoord &operator /=(T d);
	ParamCoord &operator *=(T d);

	ParamCoord<T,D> &operator*=(const Matrix<T,D+1,D+1> &m);

	using coords_base::begin;
	using coords_base::end;
};

template <class T>
struct is_point;

template <class T, int D>
struct is_point<ParamCoord<T,D>>
{
	static const bool value = true;
};

template <class T, int D> 
std::ostream &operator<<(std::ostream &out, const ParamCoord<T,D> &pt);

template <class T, int D> 
const ParamCoord<T,D> &lower(const ParamCoord<T,D> &p);

template <class T, int D, class... ARGS> 
ParamCoord<T,D> lower(const ParamCoord<T,D> &p1, const ARGS &...args);

template <class T, int D> 
const ParamCoord<T,D> &upper(const ParamCoord<T,D> &p);

template <class T, int D, class... ARGS> 
ParamCoord<T,D> upper(const ParamCoord<T,D> &p1, const ARGS &...args);

template <class T, int D> 
ParamCoord<T,D> round(const ParamCoord<T,D> p);

template <class T, int D> 
const ParamCoord<T,D> &param_origin();

}} // namespace s3d::math

namespace std
{
	template <class T, int D>
	struct make_signed<s3d::math::ParamCoord<T,D>>
	{
		typedef s3d::math::ParamCoord<typename s3d::make_signed<T>::type,D> type;
	};
	template <class T, int D>
	struct make_unsigned<s3d::math::ParamCoord<T,D>>
	{
		typedef s3d::math::ParamCoord<typename s3d::make_unsigned<T>::type,D> type;
	};
}

#include "param_coord.hpp"

#endif

// $Id: param_coord.h 3084 2010-08-31 16:08:41Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

