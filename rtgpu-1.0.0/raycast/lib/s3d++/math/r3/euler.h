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

#ifndef S3D_MATH_R3_EULER_H
#define S3D_MATH_R3_EULER_H

#include <type_traits>
#include <array>
#include "../coords.h"
#include "euler_space.h"

namespace s3d { namespace math { namespace r3
{

namespace detail/*{{{*/
{
	template <int I> struct euler_next;/*{{{*/
	template <> struct euler_next<0>
	{
		static const int value = 1;
	};
	template <> struct euler_next<1>
	{
		static const int value = 2;
	};
	template <> struct euler_next<2>
	{
		static const int value = 0;
	};
	template <> struct euler_next<3>
	{
		static const int value = 1;
	};/*}}}*/
}/*}}}*/

template <rotation_frame F>
struct rotframe_traits
{
	static const bool rotating = (F&FRAME_DYNAMICS) == FRAME_ROTATING,
					  repeating = (F&REPETITION) == REP_YES,
					  odd_parity = (F&PARITY) == PAR_ODD;

	static const rotation_frame revert = 
		static_cast<rotation_frame>((rotating?FRAME_STATIC:FRAME_ROTATING)
										| (F&~FRAME_DYNAMICS));

	static const int i = (F&AXIS) >> 3,
					 j = detail::euler_next<i+odd_parity>::value,
					 k = detail::euler_next<i+1-odd_parity>::value,
					 h = repeating ? i : k;
};

inline int main_axis(rotation_frame frame);
inline std::array<int,3> axis(rotation_frame frame);
inline bool odd_parity(rotation_frame frame);
inline bool repeating(rotation_frame frame);
inline bool rotating(rotation_frame frame);


template <rotation_frame F, class T>
struct Euler
	: coords<Euler<F,T>,euler_space<T>>
{
private:
	typedef coords<Euler,euler_space<T>> coords_base;
public:
	using coords_base::theta;
	using coords_base::phi;
	using coords_base::psi;

	static const rotation_frame frame = F;

	Euler() {}

	template <class U>
	Euler(const Euler<F,U> &that)
		: coords_base(that) {}

	Euler(const Euler &that) = default;

	// Is defined in rotation.hpp, must include it do use this ctor
	template <rotation_frame F2, class U>
	explicit Euler(const Euler<F2,U> &that);

	Euler(T theta, T phi, T psi)
		: coords_base(theta, phi, psi) {}

	bool operator==(const Euler &that) const;

	Euler operator-() const;
	Euler &operator+=(const Euler &that);
	Euler &operator-=(const Euler &that);
};

template <rotation_frame F, class T>
std::ostream &operator<<(std::ostream &out, const Euler<F,T> &r);

template <rotation_frame F, class T>
Euler<F,T> normalize(Euler<F,T> r);

template <rotation_frame F, class T>
Euler<rotframe_traits<F>::revert, T> rev(const Euler<F,T> &r);

// These should be "template typedefs" like similar types, but they aren't
// supported in gcc-4.5. Doing it this way mimicks the most common use case.

template <rotation_frame F>
Euler<F,real> euler(real theta, real phi, real psi)
{
	return {theta, phi, psi};
}

template <rotation_frame F>
Euler<F,double> deuler(double theta, double phi, double psi)
{
	return {theta, phi, psi};
}

template <rotation_frame F>
Euler<F,float> feuler(float theta, float phi, float psi)
{
	return {theta, phi, psi};
}

template <rotation_frame F>
Euler<rotframe_traits<F>::revert, real> 
	rev_euler(real theta, real phi, real psi)
{
	return {psi, phi, theta};
}

template <rotation_frame F>
Euler<rotframe_traits<F>::revert, double> 
	rev_deuler(double theta, double phi, double psi)
{
	return {psi, phi, theta};
}

template <rotation_frame F>
Euler<rotframe_traits<F>::revert, float> 
	rev_feuler(float theta, float phi, float psi)
{
	return {psi, phi, theta};
}

} // namespace r3

namespace traits
{
	template <class T>
	struct dim<r3::euler_space<T>>
	{
		typedef mpl::vector_c<int,3> type;
	};
};

using r3::euler;
using r3::deuler;
using r3::feuler;

using r3::rev_euler;
using r3::rev_deuler;
using r3::rev_feuler;

}} // namespace s3d::math

#include "euler.hpp"

#endif

// $Id: euler.h 3143 2010-09-21 18:36:05Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

