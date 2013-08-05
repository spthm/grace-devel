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

#include "real.h"

namespace s3d { namespace math
{

// result_* stuff
template <class U, int D1, class T, int D2>
struct result_add<ParamCoord<U,D1>, ParamCoord<T,D2>> /*{{{*/
{
	static_assert(D1==RUNTIME || D2==RUNTIME || D1==D2, "Mismatched dimension");

	typedef ParamCoord<typename result_add_dispatch<U,T>::type, 
					   D1==RUNTIME?D2:D1> type;
};/*}}}*/
template <class U, int D1, class T, int D2>
struct result_sub<ParamCoord<U,D1>, ParamCoord<T,D2>> /*{{{*/
{
	static_assert(D1==RUNTIME || D2==RUNTIME || D1==D2, "Mismatched dimension");

	typedef ParamCoord<typename result_sub_dispatch<U,T>::type, 
					   D1==RUNTIME?D2:D1> type;
};/*}}}*/
template <class U, int D1, class T, int D2>
struct result_mul<ParamCoord<U,D1>, ParamCoord<T,D2>> /*{{{*/
{
	static_assert(D1==RUNTIME || D2==RUNTIME || D1==D2, "Mismatched dimension");

	typedef ParamCoord<typename result_mul_dispatch<U,T>::type, 
					   D1==RUNTIME?D2:D1> type;
};/*}}}*/
template <class U, int D1, class T, int D2>
struct result_div<ParamCoord<U,D1>, ParamCoord<T,D2>> /*{{{*/
{
	static_assert(D1==RUNTIME || D2==RUNTIME || D1==D2, "Mismatched dimension");

	typedef ParamCoord<typename result_div_dispatch<U,T>::type, 
					   D1==RUNTIME?D2:D1> type;
};/*}}}*/

#if 0
template <class T, int N, class U>
struct result_add<ParamCoord<T,N>, U>/*{{{*/
{
	static_assert(order<ParamCoord<T,N>>::value == order<U>::value, "Wrong order");

	typedef ParamCoord<typename result_add<T,U>::type, N> type;
};/*}}}*/
template <class T, int N, class U>
struct result_sub<ParamCoord<T,N>, U>/*{{{*/
{
	static_assert(order<ParamCoord<T,N>>::value == order<U>::value, "Wrong order");

	typedef ParamCoord<typename result_sub<T,U>::type, N> type;
};/*}}}*/
template <class T, int N, class U>
struct result_mul<ParamCoord<T,N>, U>/*{{{*/
{
	static_assert(order<ParamCoord<T,N>>::value == order<U>::value, "Wrong order");

	typedef ParamCoord<typename result_mul<T,U>::type, N> type;
};/*}}}*/
template <class T, int N, class U>
struct result_div<ParamCoord<T,N>, U>/*{{{*/
{
	static_assert(order<ParamCoord<T,N>>::value == order<U>::value, "Wrong order");

	typedef ParamCoord<typename result_div<T,U>::type, N> type;
};/*}}}*/
#endif

template <class T, int D> 
ParamCoord<T,D>::ParamCoord(const T &v)/*{{{*/
{
	std::fill(begin(), end(), v);
}/*}}}*/
template <class T, int D> template <class U> 
ParamCoord<T,D>::ParamCoord(const ParamCoord<U,D> &that)/*{{{*/
{
	std::copy(that.begin(), that.end(), begin());
}/*}}}*/
template <class T, int D> 
ParamCoord<T,D>::ParamCoord(const ParamCoord &that)/*{{{*/
	: coords_base(that)
{
}/*}}}*/

template <class T, int D> template <class U> 
ParamCoord<T,D>::ParamCoord(const Point<U,D> &that)/*{{{*/
{
	std::copy(that.begin(), that.end(), begin());
}/*}}}*/

template <class T, int D>
ParamCoord<T,D>::operator Point<T,D>() const/*{{{*/
{
	Point<T,D> p;
	std::copy(begin(), end(), p.begin());
	return p;
}/*}}}*/

template <class T, int D> 
ParamCoord<T,D> &ParamCoord<T,D>::operator+=(const Vector<T,D> &v)/*{{{*/
{
	auto it1 = begin(); auto it2 = v.begin();
	while(it1 != end())
		*it1++ += *it2++;
	return *this;
}/*}}}*/
template <class T, int D> 
ParamCoord<T,D> &ParamCoord<T,D>::operator-=(const Vector<T,D> &v)/*{{{*/
{
	auto it1 = begin(); auto it2 = v.begin();
	while(it1 != end())
		*it1++ -= *it2++;
	return *this;
}/*}}}*/

template <class T, int D> 
ParamCoord<T,D> &ParamCoord<T,D>::operator*=(const Size<T,D> &s)/*{{{*/
{
	auto it1 = begin(); auto it2 = s.begin();
	while(it1 != end())
		*it1++ *= *it2++;
	return *this;
}/*}}}*/
template <class T, int D> 
ParamCoord<T,D> &ParamCoord<T,D>::operator/=(const Size<T,D> &s)/*{{{*/
{
	auto it1 = begin(); auto it2 = s.begin();
	while(it1 != end())
		*it1++ /= *it2++;
	return *this;
}/*}}}*/

template <class T, int D> 
Vector<T,D> ParamCoord<T,D>::operator-(const ParamCoord &p) const/*{{{*/
{
	Vector<T,D> v;

	auto itp = p.begin(); auto itv = v.begin();

	for(auto it=begin(); it!=end(); ++it, ++itv, ++itp)
		*itv = *it - *itp;
	return v;
}/*}}}*/

template <class T, int D> 
ParamCoord<T,D> &ParamCoord<T,D>::operator +=(T d)/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		*it += d;
	return *this;
}/*}}}*/
template <class T, int D> 
ParamCoord<T,D> &ParamCoord<T,D>::operator -=(T d)/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		*it -= d;
	return *this;
}/*}}}*/
template <class T, int D> 
ParamCoord<T,D> &ParamCoord<T,D>::operator /=(T d)/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		*it /= d;
	return *this;
}/*}}}*/
template <class T, int D> 
ParamCoord<T,D> &ParamCoord<T,D>::operator *=(T d)/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		*it *= d;
	return *this;
}/*}}}*/

template <class T, int D> 
bool ParamCoord<T,D>::operator==(const ParamCoord &that) const/*{{{*/
{
	auto it1 = begin(); auto it2 = that.begin();
	while(it1 != end())
		if(!equal(*it1++, *it2++))
			return false;
	return true;
}/*}}}*/

template <class T, int D> 
const ParamCoord<T,D> &lower(const ParamCoord<T,D> &p)/*{{{*/
{
	return p;
}/*}}}*/
template <int D, class... ARGS, class T> 
ParamCoord<T,D> lower(const ParamCoord<T,D> &p1,/*{{{*/
														  const ARGS &...args)
{
	ParamCoord<T,D> p2 = lower(args...), l;

	auto it1 = p1.begin();
	auto it2 = p2.begin();
	auto itr = l.begin();

	while(it1 != p1.end())
		*itr++ = min(*it1++, *it2++);

	return l;
}/*}}}*/

template <class T, int D> 
const ParamCoord<T,D> &upper(const ParamCoord<T,D> &p)/*{{{*/
{
	return p;
}/*}}}*/
template <int D, class... ARGS, class T> 
ParamCoord<T,D> upper(const ParamCoord<T,D> &p1, const ARGS &...args)/*{{{*/
{
	ParamCoord<T,D> p2 = upper(args...), u;

	auto it1 = p1.begin();
	auto it2 = p2.begin();
	auto itr = u.begin();

	while(it1 != p1.end())
		*itr++ = max(*it1++, *it2++);

	return u;
}/*}}}*/

template <class T, int D> 
ParamCoord<T,D> round(const ParamCoord<T,D> p)/*{{{*/
{
	Point<T,D> out(dim(p.size()));

	auto it1 = out.begin(); auto it2 = p.begin();

	while(it1 != out.end())
		*it1++ = round(*it2++);
	return out;
}/*}}}*/

template <class T, int D> 
const ParamCoord<T,D> &param_origin()/*{{{*/
{
	ParamCoord<T,D> o(T(0));
	return o;
}/*}}}*/

template <class T, int D> 
std::ostream &operator<<(std::ostream &out, const ParamCoord<T,D> &p)/*{{{*/
{
	out << "<"; 
	for(int i=0; i<D; ++i)
	{
		out << p[i];
		if(i < D-1)
			out << ';';
	}
	return out << ">";
}/*}}}*/

}} // namespace s3d::math

// $Id: param_coord.hpp 2972 2010-08-18 05:22:05Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

