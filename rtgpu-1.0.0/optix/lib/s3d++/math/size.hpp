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

#include <cassert>
#include <boost/functional/hash.hpp>
#include "real.h"
#include <limits>
#include <algorithm>
#include "operators.h"

namespace s3d { namespace math
{
// constructors

template <class T, int D> 
Size<T,D>::Size(const T &v)/*{{{*/
{
	std::fill(begin(), end(), v);
}/*}}}*/
template <class T, int D> template <class U> 
Size<T,D>::Size(const Size<U,D> &that)/*{{{*/
{
	std::copy(that.begin(), that.end(), begin());
}/*}}}*/
template <class T, int D> 
Size<T,D>::Size(const Size &that)/*{{{*/
	: coords_base(that)
{
}/*}}}*/

template <class T, int N> 
Size<T,N>::Size(const dimension<1>::type &d, const T &v)/*{{{*/
	: coords_base(d)
{
	std::fill(begin(), end(), v);
}/*}}}*/

// arithmetic operators
template <class T, int D> 
Size<T,D> Size<T,D>::operator -()/*{{{*/
{
	Size s = *this;
	std::transform(s.begin(), s.end(), s.begin(), std::negate<T>());
	return s;
}/*}}}*/

template <class T, int D> template <class U, int M, class>
auto Size<T,D>::operator+=(const Size<U,M> &s) -> Size &/*{{{*/
{
	if(size() != s.size())
		throw std::runtime_error("Mismatched size dimensions");

	auto it1 = begin(); auto it2 = s.begin();
	while(it1 != end())
		*it1++ += *it2++;
	return *this;
}/*}}}*/
template <class T, int D> template <class U, int M, class>
auto Size<T,D>::operator-=(const Size<U,M> &s) -> Size &/*{{{*/
{
	if(size() != s.size())
		throw std::runtime_error("Mismatched size dimensions");

	auto it1 = begin(); auto it2 = s.begin();
	while(it1 != end())
		*it1++ -= *it2++;
	return *this;
}/*}}}*/
template <class T, int D> template <class U, int M, class>
auto Size<T,D>::operator*=(const Size<U,M> &s) -> Size &/*{{{*/
{
	if(size() != s.size())
		throw std::runtime_error("Mismatched size dimensions");

	auto it1 = begin(); auto it2 = s.begin();
	while(it1 != end())
		*it1++ *= *it2++;
	return *this;
}/*}}}*/
template <class T, int D> template <class U, int M, class>
auto Size<T,D>::operator/=(const Size<U,M> &s) -> Size &/*{{{*/
{
	if(size() != s.size())
		throw std::runtime_error("Mismatched size dimensions");

	auto it1 = begin(); auto it2 = s.begin();
	while(it1 != end())
		*it1++ /= *it2++;
	return *this;
}/*}}}*/

template <class T, int D> template <class U, int M, class>
auto Size<T,D>::operator+=(const Vector<U,M> &s) -> Size &/*{{{*/
{
	if(size() != s.size())
		throw std::runtime_error("Mismatched size dimensions");

	auto it1 = begin(); auto it2 = s.begin();
	while(it1 != end())
		*it1++ += *it2++;
	return *this;
}/*}}}*/
template <class T, int D> template <class U, int M, class>
auto Size<T,D>::operator-=(const Vector<U,M> &s) -> Size &/*{{{*/
{
	if(size() != s.size())
		throw std::runtime_error("Mismatched size dimensions");

	auto it1 = begin(); auto it2 = s.begin();
	while(it1 != end())
		*it1++ -= *it2++;
	return *this;
}/*}}}*/

template <class T, int D> 
Size<T,D> &Size<T,D>::operator +=(T d)/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		*it += d;
	return *this;
}/*}}}*/
template <class T, int D> 
Size<T,D> &Size<T,D>::operator -=(T d)/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		*it -= d;
	return *this;
}/*}}}*/
template <class T, int D> 
Size<T,D> &Size<T,D>::operator /=(T d)/*{{{*/
{
#if 0
	if(d == 0)
		std::fill(begin(), end(), std::numeric_limits<T>::max());
	else
	{
#endif
		for(auto it=begin(); it!=end(); ++it)
			*it /= d;
#if 0
	}
#endif
	return *this;
}/*}}}*/
template <class T, int D>
Size<T,D> &Size<T,D>::operator *=(T d)/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		*it *= d;
	return *this;
}/*}}}*/
template <class T1, class T2, int D1, int D2>
auto operator==(const Size<T1,D1> &s1, const Size<T2,D2> &s2)/*{{{*/
	-> typename std::enable_if<D1==D2 || D1==RUNTIME || D2==RUNTIME, bool>::type
{
	if(s1.size() != s2.size())
		return false;

	auto it1 = s1.begin(); auto it2 = s2.begin();
	while(it1 != s1.end())
		if(!equal(*it1++, *it2++))
			return false;
	return true;
}/*}}}*/

template <class T, int D> 
bool Size<T,D>::is_positive() const/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		if(less_than(*it,0))
			return false;
	return true;
}/*}}}*/
template <class T, int D> 
bool Size<T,D>::is_negative() const/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		if(greater_than(*it,0))
			return false;
	return true;
}/*}}}*/
template <class T, int D> 
bool Size<T,D>::is_zero() const/*{{{*/
{
	for(auto it=begin(); it!=end(); ++it)
		if(!equal(*it,0))
			return false;
	return true;
}/*}}}*/

template <class T, int D> 
T max_dim(const Size<T,D> &s)/*{{{*/
{
	auto it = s.begin();
	T cur_max = *it;

	for(++it; it!=s.end(); ++it)
		cur_max = max(cur_max, *it);

	return cur_max;
}/*}}}*/

template <class T, int D> 
T min_dim(const Size<T,D> &s)/*{{{*/
{
	auto it = s.begin();
	T cur_max = *it;

	for(++it; it!=s.end(); ++it)
		cur_max = min(cur_max, *it);

	return cur_max;
}/*}}}*/

template <class T, int D> 
const Size<T,D> &lower(const Size<T,D> &p)/*{{{*/
{
	return p;
}/*}}}*/
template <class T, int D, class... ARGS> 
Size<T,D> lower(const Size<T,D> &p1, const ARGS &...args)/*{{{*/
{
	static_assert(D!=RUNTIME, "Function doesn't work with dynamic sizes");

	Size<T,D> p2 = lower(args...), l;

	auto it1 = p1.begin();
	auto it2 = p2.begin();
	auto itr = l.begin();

	while(it1 != p1.end())
		*itr++ = min(*it1++, *it2++);
	return l;
}/*}}}*/

template <class T, int D> 
const Size<T,D> &upper(const Size<T,D> &p)/*{{{*/
{
	return p;
}/*}}}*/
template <class T, int D, class... ARGS> 
Size<T,D> upper(const Size<T,D> &p1, const ARGS &...args)/*{{{*/
{
	static_assert(D!=RUNTIME, "Function doesn't work with dynamic sizes");

	Size<T,D> p2 = upper(args...), u;

	auto it1 = p1.begin();
	auto it2 = p2.begin();
	auto itr = u.begin();

	while(it1 != p1.end())
		*itr++ = max(*it1++, *it2++);
	return u;
}/*}}}*/

template <class T, int D> 
Size<T,D> round(const Size<T,D> &p)/*{{{*/
{
	using std::round;
	Size<T,D> out;

	auto it1 = out.begin(); auto it2 = p.begin();
	while(it1 != out.end())
		*it1++ = round(*it2++);
	return out;
}/*}}}*/

template <class T, int D> 
Size<T,D> ceil(const Size<T,D> &p)/*{{{*/
{
	using std::ceil;
	Size<T,D> out;
	for(std::size_t i=0; i<p.size(); ++i)
		out[i] = ceil(p[i]);
	return out;
}/*}}}*/

template <class T, int D> 
Size<T,D> floor(const Size<T,D> &p)/*{{{*/
{
	using std::floor;

	Size<T,D> out;
	for(std::size_t i=0; i<p.size(); ++i)
		out[i] = floor(p[i]);
	return out;
}/*}}}*/

template <class T, int D> 
std::ostream &operator<<(std::ostream &out, const Size<T,D> &sz)/*{{{*/
{
	out << "("; 
	for(std::size_t i=0; i<sz.size(); ++i)
	{
		out << sz[i];
		if(i < D-1)
			out << 'x';
	}
	return out << ")";
}/*}}}*/
template <class T, int D> 
std::istream &operator>>(std::istream &in, Size<T,D> &sz)/*{{{*/
{
	if(in.peek() == '(')
		in.ignore();

	for(std::size_t i=0; i<sz.size(); ++i)
	{
		in >> sz[i];
		if(i < D-1 && (in.peek() == 'x' || in.peek() == ';' || in.peek()==','))
			in.ignore();
	}

	if(!in.eof() && in.peek() == ')')
		in.ignore();
	return in;
}/*}}}*/

template <class T, int D>
size_t hash_value(const Size<T,D> &p)/*{{{*/
{
	return boost::hash_range(p.begin(), p.end());
}/*}}}*/

template <class T, int D>
T area(const Size<T,D> &s)/*{{{*/
{
	T a = 1;
	for(auto it=s.begin(); it!=s.end() && a!=0; ++it)
		a *= *it;
	return a;
}/*}}}*/

template <class T>
Size<T,2> &transpose_inplace(Size<T,2> &s)/*{{{*/
{
	swap(s.w, s.h);
	return s;
}/*}}}*/

template <class T>
Size<T,2> &&transpose(Size<T,2> &&s)/*{{{*/
{
	return std::move(transpose_inplace(s));
};/*}}}*/

template <class T>
Size<T,2> transpose(const Size<T,2> &s)/*{{{*/
{
	return transpose(Size<T,2>(s));
}/*}}}*/

template <class V>
auto inv(V &&v)/*{{{*/
	-> typename std::enable_if<!std::is_const<typename std::remove_reference<V>::type>::value &&
							   is_size<V>::value, V &&>::type
{
	for(auto it=v.begin(); it!=v.end(); ++it)
		*it = 1 / *it;
	return std::forward<V>(v);
}/*}}}*/

template <class V>
auto inv(const V &v)/*{{{*/
	-> typename std::enable_if<is_size<V>::value, V>::type
{
	return inv(V(v));
}/*}}}*/


}} // namespace s3d::math

// $Id: size.hpp 3132 2010-09-15 00:18:39Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

