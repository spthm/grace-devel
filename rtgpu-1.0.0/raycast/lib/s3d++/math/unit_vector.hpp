
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

#include "operators.h"
#include "../util/type_traits.h"

namespace s3d { namespace math
{

// Result type of arithmetic operations {{{
template <class T, int N, class U, int P>
struct result_add<UnitVector<T,N>, UnitVector<U,P>>
	: result_add_dispatch<Vector<T,N>, Vector<U,P>>
{
};

template <class T, int N, class U, int P>
struct result_sub<UnitVector<T,N>, UnitVector<U,P>> 
	: result_sub_dispatch<Vector<T,N>, Vector<U,P>>
{
};

template <class T, int N, class U, int P>
struct result_mul<UnitVector<T,N>, UnitVector<U,P>>
	: result_mul_dispatch<Vector<T,N>, Vector<U,P>>
{
};

template <class T, int N, class U, int P>
struct result_div<UnitVector<T,N>, UnitVector<U,P>>
	: result_div_dispatch<Vector<T,N>, Vector<U,P>>
{
};

template <class T, int M, int N, class U, int P>
struct result_mul<Matrix<T,M,N>,UnitVector<U,P>> 
	: result_mul_dispatch<Matrix<T,M,N>, Vector<U,N>>
{
};

template <class T, int N, class U>
struct result_add<UnitVector<T,N>, arithmetic<U>> 
	: result_add_dispatch<Vector<T,N>,U>
{
};

template <class T, int N, class U>
struct result_sub<UnitVector<T,N>, arithmetic<U>> 
	: result_sub_dispatch<Vector<T,N>,U>
{
};

template <class T, int N, class U>
struct result_mul<UnitVector<T,N>, arithmetic<U>> 
	: result_mul_dispatch<Vector<T,N>,U>
{
};

template <class T, int N, class U>
struct result_div<UnitVector<T,N>, arithmetic<U>> 
	: result_div_dispatch<Vector<T,N>,U>
{
};

template <class T, class U, int N>
struct result_add<arithmetic<T>, UnitVector<U,N>> 
	: result_add_dispatch<T,Vector<U,N>>
{
};

template <class T, class U, int N>
struct result_sub<arithmetic<T>, UnitVector<U,N>> 
	: result_sub_dispatch<T,Vector<U,N>>
{
};

template <class T, class U, int N>
struct result_mul<arithmetic<T>, UnitVector<U,N>> 
	: result_mul_dispatch<T,Vector<U,N>>
{
};

template <class T, class U, int N>
struct result_div<arithmetic<T>, UnitVector<U,N>> 
	: result_div_dispatch<T,Vector<U,N>>
{
};/*}}}*/

template <class T, int D, class U=typename make_floating_point<T>::type>
auto unit(const Vector<T,D> &c, U *_norm=NULL)/*{{{*/
	-> UnitVector<typename make_floating_point<T>::type,D>
{
	auto n2 = sqrnorm(c);
	assert(n2 != 0);

	if(equal(n2,1))
	{
		if(_norm)
			*_norm = 1;

		return { c, true };
	}
	else
	{
		auto n = sqrt(n2);

		if(_norm)
			*_norm = n;

		return { c / n, true };
	}
}/*}}}*/

template <int D, int M=0, class T=real> 
auto axis() -> UnitVector<T,D>/*{{{*/
{
	Vector<T,D> a;
	for(int i=0; i<D; ++i)
		a[i] = i==M ? 1 : 0;

	return UnitVector<T,D>{a, true};
}/*}}}*/

template <class T, int D> 
UnitVector<T,D>::UnitVector()/*{{{*/
	: coords_base(axis<D,0,T>())
{
}/*}}}*/

template <class T, int D> 
UnitVector<T,D>::UnitVector(const Vector<T,D> &c)/*{{{*/
	: coords_base(unit(c))
{
}/*}}}*/

template <class T, int D> template <class... ARGS, class>
UnitVector<T,D>::UnitVector(T c1, ARGS... cn)/*{{{*/
	: coords_base(unit(Vector<T,D>(c1, cn...)))
{
}/*}}}*/

template <class T, int D> template <class U>
UnitVector<T,D>::operator Vector<U,D>() const/*{{{*/
{
	Vector<U,D> v(this->size());
	std::copy(begin(), end(), v.begin());
	return v;
}/*}}}*/

template <class T, int D> 
UnitVector<T,D>::UnitVector(const Vector<T,D> &c, bool is_unit)/*{{{*/
	: coords_base(is_unit ? reinterpret_cast<const UnitVector &>(c) : unit(c))
{
	assert(math::is_unit(static_cast<const Vector<T,D>&>(*this)));
}/*}}}*/


template <class T, int D> 
UnitVector<T,D> &UnitVector<T,D>::operator=(const UnitVector<T,D> &that)/*{{{*/
{
	coords_base::operator=(that);
	return *this;
}/*}}}*/

template <class T, int D> 
auto UnitVector<T,D>::operator-() const -> UnitVector/*{{{*/
{
	UnitVector v = *this;
	for(auto it = v.begin(); it!=v.end(); ++it)
		const_cast<T&>(*it) = -*it;
	return v;
}/*}}}*/

template <class T, int D>
auto UnitVector<T,D>::operator*=(const UnitVector &that) -> UnitVector &/*{{{*/
{
	reinterpret_cast<Vector<T,D> &>(*this) *= that;
	return *this;
}/*}}}*/

template <class T, int D>
auto UnitVector<T,D>::operator/=(const UnitVector &that) -> UnitVector &/*{{{*/
{
	reinterpret_cast<Vector<T,D> &>(*this) /= that;
	return *this;
}/*}}}*/

template <class T, int D> const UnitVector<T,D> &unit(const UnitVector<T,D> &c)/*{{{*/
{
	return c;
}/*}}}*/
template <class T, int D> bool is_unit(const UnitVector<T,D> &c)/*{{{*/
{
	return true;
}/*}}}*/

template <class A=real, class T, int D> 
A angle(const UnitVector<T,D> &v1, const UnitVector<T,D> &v2)/*{{{*/
{
	// Retorna de 0 a Pi radianos
	T d = dot(v1, v2);
	if(equal(d,1)) // when d is almost 1, acos is returning NaN
		return 0;
	else if(equal(d,-1))
		return M_PI;
	else
		return acos(d);
}/*}}}*/

template <class T, int D> T abs(const UnitVector<T,D> &c)/*{{{*/
{
	return 1;
}/*}}}*/
template <class T, int D> T norm(const UnitVector<T,D> &c)/*{{{*/
{
	return 1;
}/*}}}*/
template <class T, int D> T sqrnorm(const UnitVector<T,D> &c)/*{{{*/
{
	return 1;
}/*}}}*/

template <class T, int N, class... ARGS> 
auto augment(const UnitVector<T,N> &p, const ARGS &...c)/*{{{*/
	-> Vector<typename std::common_type<T,ARGS...>::type, N+sizeof...(ARGS)>
{
	return concat(Vector<T,N>(p), 
				  Vector<typename std::common_type<ARGS...>::type,
							sizeof...(ARGS)>(c...));
}/*}}}*/

}} // namespace s3d::math

// $Id: complex.hpp 2752 2010-06-11 02:32:41Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

