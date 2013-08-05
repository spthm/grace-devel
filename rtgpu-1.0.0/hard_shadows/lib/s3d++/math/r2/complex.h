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

#ifndef S3D_MATH_R2_COMPLEX_H
#define S3D_MATH_R2_COMPLEX_H

#include <array>
#include "../../util/concepts.h"
#include "../real.h"
#include "../coords.h"
#include "complex_space.h"
#include "fwd.h"

// I still hate macros
#ifdef Complex
#undef Complex
#endif

namespace s3d { namespace math { namespace r2
{
template <class T>
struct is_complex
{
	static const bool value = false;
};

template <class T> struct Complex
	: coords<Complex<T>,complex_space<T>>
{
private:
	typedef coords<Complex,complex_space<T>> coords_base;
public:
	using coords_base::re;
	using coords_base::im;

	template <class U> struct rebind { typedef Complex<U> type; };

	Complex() {}

	Complex(T re, T im=0) : coords_base(re,im) {}

	template <class U>
	Complex(const Complex<U> &c) : coords_base(c) {}

private:
#if HAS_SERIALIZATION
	friend class boost::serialization::access;

	template <class A>
	void serialize(A &ar, unsigned int version)
	{
		ar & re;
		ar & im;
	}
#endif
};

template <class T>
struct is_complex<Complex<T>>
{
	static const bool value = true;
};

} // namespace r2

namespace traits
{
	template <class T>
	struct dim<r2::complex_space<T>>
	{
		typedef mpl::vector_c<int, 2> type;
	};
}

}} // namespace s3d::math::r2

namespace std
{

	template <class T>
	struct make_signed<s3d::math::r2::Complex<T>>
	{
		typedef s3d::math::r2::Complex<typename make_signed<T>::type> type;
	};
	template <class T>
	struct make_unsigned<s3d::math::r2::Complex<T>>
	{
		typedef s3d::math::r2::Complex<typename make_unsigned<T>::type> type;
	};
}

#include "complex.hpp"

#endif

// $Id: complex.h 3085 2010-08-31 18:32:10Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

