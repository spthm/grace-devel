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

#include <boost/functional/hash.hpp>
#include "detail.h"
#include "real.h"

namespace s3d { namespace math
{

// Result types {{{

template <class T, class U>
struct result_sub<point_like<T>,point_like<U>>/*{{{*/
{
	static_assert(detail::equaldim<T,U>::value, "Mismatched dimension");
	
	typedef typename result_sub_dispatch<typename value_type<T>::type,
										 typename value_type<U>::type>::type
				subtype;

	typedef Vector<subtype, T::dim==RUNTIME?U::dim:T::dim> type;

	static inline type call(const T &lhs, const T &rhs) 
	{
		type ret(lhs.size());

		auto itl=lhs.begin(); auto itr=rhs.begin();
		for(auto it=ret.begin(); it!=ret.end(); ++it, ++itl, ++itr)
			*it = *itl - *itr;

		return ret;
	};
};/*}}}*/

template <class P, class V>
struct result_add<point_like<P>,vector_like<V>>/*{{{*/
{
	static_assert(detail::equaldim<P,V>::value, "Mismatched dimension");

	typedef typename result_add_dispatch<typename value_type<P>::type,
										 typename value_type<V>::type>::type
				subtype;

	typedef typename rebind<P,subtype>::type type;
};/*}}}*/

template <class P, class V>
struct result_add<vector_like<V>,point_like<P>>
	: result_add<point_like<P>, vector_like<V>>
{
	static const bool revert = true;
};

template <class P, class V>
struct result_sub<point_like<P>,vector_like<V>>/*{{{*/
{
	static_assert(detail::equaldim<P,V>::value, "Mismatched dimension");

	typedef typename result_sub_dispatch<typename value_type<P>::type,
										 typename value_type<V>::type>::type
				subtype;

	typedef typename rebind<P,subtype>::type type;
};/*}}}*/

template <class P, class V>
struct result_sub<vector_like<V>,point_like<P>>
	: result_sub<point_like<P>, vector_like<V>>
{
	static const bool revert = true;
};

template <class P, class S>
struct result_add<point_like<P>,size_like<S>>/*{{{*/
{
	static_assert(detail::equaldim<P,S>::value, "Mismatched dimension");

	typedef typename result_add_dispatch<typename value_type<P>::type,
										 typename value_type<S>::type>::type
				subtype;

	typedef typename rebind<P,subtype>::type type;
};/*}}}*/

template <class S, class P>
struct result_add<size_like<S>,point_like<P>>
	: result_add<point_like<P>, size_like<S>>
{
	static const bool revert = true;
};

template <class P, class S>
struct result_sub<point_like<P>,size_like<S>>/*{{{*/
{
	static_assert(detail::equaldim<P,S>::value, "Mismatched dimension");

	typedef typename result_sub_dispatch<typename value_type<P>::type,
										 typename value_type<S>::type>::type
				subtype;

	typedef typename rebind<P,subtype>::type type;
};/*}}}*/

template <class S, class P>
struct result_sub<size_like<S>, point_like<P>>
	: result_sub<point_like<P>, size_like<S>>
{
	static const bool revert = true;
};

//}}}

template <class T, int N> template <class DUMMY, class>
Point<T,N>::Point(const T &v)/*{{{*/
{
	std::fill(begin(), end(), v);
}/*}}}*/
template <class T, int N> 
Point<T,N>::Point(const dimension<1>::type &d, const T &v)/*{{{*/
	: coords_base(d)
{
	std::fill(begin(), end(), v);
}/*}}}*/
template <class T, int D> 
template <class U> Point<T,D>::Point(const Point<U,D> &that)/*{{{*/
{
	std::copy(that.begin(), that.end(), begin());
}/*}}}*/
template <class T, int D> 
Point<T,D>::Point(const Point &that)/*{{{*/
	: coords_base(that)
{
}/*}}}*/

template <class T, int D>
Point<T,D>::Point(const Vector<T,D> &v)/*{{{*/
	: coords_base(v)
{
//	std::fill(v.begin(), v.end(), begin());
}/*}}}*/

template <class T, int D>
Vector<T,D>::Vector(const Point<T,D> &p)/*{{{*/
	: coords_base(p)
{
//	std::fill(p.begin(), p.end(), begin());
}/*}}}*/

template <class V>
auto operator -(const V &v)/*{{{*/
	-> typename std::enable_if<is_point<V>::value, V>::type
{
	V r(dim(v.size()));
	std::transform(v.begin(), v.end(), r.begin(), 
				   std::negate<typename V::value_type>());
	return r;
}/*}}}*/

template <class P, class V>
auto operator+=(P &&p, const V &v)/*{{{*/
	-> typename std::enable_if<is_point<P>::value && is_vector<V>::value, 
				P&&>::type
{
	static_assert(detail::equaldim<P,V>::value, "Mismatched dimensions");

	if(p.size() != v.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = p.begin(); auto it2 = v.begin();
	while(it1 != p.end())
		*it1++ += *it2++;

	return std::forward<P>(p);
}/*}}}*/
template <class P, class V>
auto operator-=(P &&p, const V &v)/*{{{*/
	-> typename std::enable_if<is_point<P>::value && is_vector<V>::value, 
				P&&>::type
{
	static_assert(detail::equaldim<P,V>::value, "Mismatched dimensions");

	if(p.size() != v.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = p.begin(); auto it2 = v.begin();
	while(it1 != p.end())
		*it1++ -= *it2++;

	return std::forward<P>(p);
}/*}}}*/

template <class P, class S>
auto operator+=(P &&p, const S &s)/*{{{*/
	-> typename std::enable_if<is_point<P>::value && is_size<S>::value, 
				P&&>::type
{
	static_assert(detail::equaldim<P,S>::value, "Mismatched dimensions");

	if(p.size() != s.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = p.begin(); auto it2 = s.begin();
	while(it1 != p.end())
		*it1++ += *it2++;

	return std::forward<P>(p);
}/*}}}*/
template <class P, class S>
auto operator-=(P &&p, const S &s)/*{{{*/
	-> typename std::enable_if<is_point<P>::value && is_size<S>::value, 
				P&&>::type
{
	static_assert(detail::equaldim<P,S>::value, "Mismatched dimensions");

	if(p.size() != s.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = p.begin(); auto it2 = s.begin();
	while(it1 != p.end())
		*it1++ -= *it2++;

	return std::forward<P>(p);
}/*}}}*/
template <class P, class S>
auto operator*=(P &&p, const S &s)/*{{{*/
	-> typename std::enable_if<is_point<P>::value && is_size<S>::value, 
				P&&>::type
{
	static_assert(detail::equaldim<P,S>::value, "Mismatched dimensions");

	if(p.size() != s.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = p.begin(); auto it2 = s.begin();
	while(it1 != p.end())
		*it1++ *= *it2++;

	return std::forward<P>(p);
}/*}}}*/
template <class P, class S>
auto operator/=(P &&p, const S &s)/*{{{*/
	-> typename std::enable_if<is_point<P>::value && is_size<S>::value, 
				P&&>::type
{
	static_assert(detail::equaldim<P,S>::value, "Mismatched dimensions");

	if(p.size() != s.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = p.begin(); auto it2 = s.begin();
	while(it1 != p.end())
		*it1++ /= *it2++;

	return std::forward<P>(p);
}/*}}}*/

template <class P>
auto operator +=(P &&p, typename value_type<P>::type d)/*{{{*/
	-> typename std::enable_if<is_point<P>::value, P &&>::type
{
	for(auto it=p.begin(); it!=p.end(); ++it)
		*it += d;
	return std::forward<P>(p);
}/*}}}*/
template <class P>
auto operator -=(P &&p, typename value_type<P>::type d)/*{{{*/
	-> typename std::enable_if<is_point<P>::value, P &&>::type
{
	for(auto it=p.begin(); it!=p.end(); ++it)
		*it -= d;
	return std::forward<P>(p);
}/*}}}*/
template <class P>
auto operator *=(P &&p, typename value_type<P>::type d)/*{{{*/
	-> typename std::enable_if<is_point<P>::value, P &&>::type
{
	for(auto it=p.begin(); it!=p.end(); ++it)
		*it *= d;
	return std::forward<P>(p);
}/*}}}*/
template <class P>
auto operator /=(P &&p, typename value_type<P>::type d)/*{{{*/
	-> typename std::enable_if<is_point<P>::value, P &&>::type
{
	for(auto it=p.begin(); it!=p.end(); ++it)
		*it /= d;
	return std::forward<P>(p);
}/*}}}*/

template <class T, int D> 
const Point<T,D> &lower(const Point<T,D> &p)/*{{{*/
{
	return p;
}/*}}}*/
template <class T, int D, class... ARGS> 
Point<T,D> lower(const Point<T,D> &p1, const ARGS &...args)/*{{{*/
{
	Point<T,D> p2 = lower(args...), l;

	auto it1 = p1.begin();
	auto it2 = p2.begin();
	auto itr = l.begin();

	while(it1 != p1.end())
		*itr++ = min(*it1++, *it2++);

	return l;
}/*}}}*/

template <class T, int D> 
const Point<T,D> &upper(const Point<T,D> &p)/*{{{*/
{
	return p;
}/*}}}*/
template <class T, int D, class... ARGS> 
Point<T,D> upper(const Point<T,D> &p1, const ARGS &...args)/*{{{*/
{
	Point<T,D> p2 = upper(args...), u;

	auto it1 = p1.begin();
	auto it2 = p2.begin();
	auto itr = u.begin();

	while(it1 != p1.end())
		*itr++ = max(*it1++, *it2++);

	return u;
}/*}}}*/

template <class T, int D> 
Point<T,D> round(const Point<T,D> &p)/*{{{*/
{
	Point<T,D> out(dim(p.size()));

	auto it1 = out.begin(); auto it2 = p.begin();

	while(it1 != out.end())
		*it1++ = round(*it2++);
	return out;
}/*}}}*/

template <class T, int D> 
Point<T,D> rint(const Point<T,D> &p)/*{{{*/
{
	Point<T,D> out(dim(p.size()));

	auto it1 = out.begin(); auto it2 = p.begin();

	while(it1 != out.end())
		*it1++ = rint(*it2++);
	return out;
}/*}}}*/

template <class T, int D> 
Point<T,D> ceil(const Point<T,D> &p)/*{{{*/
{
	Point<T,D> out(dim(p.size()));

	auto it1 = out.begin(); auto it2 = p.begin();

	while(it1 != out.end())
		*it1++ = ceil(*it2++);
	return out;
}/*}}}*/

template <class T, int D> 
Point<T,D> floor(const Point<T,D> &p)/*{{{*/
{
	Point<T,D> out(dim(p.size()));

	auto it1 = out.begin(); auto it2 = p.begin();

	while(it1 != out.end())
		*it1++ = floor(*it2++);
	return out;
}/*}}}*/

template <class T, int D> 
Point<T,D> origin()/*{{{*/
{
	Point<T,D> o(T(0));
	return o;
}/*}}}*/

template <class T, int D> 
std::ostream &operator<<(std::ostream &out, const Point<T,D> &p)/*{{{*/
{
	out << "("; 
	for(int i=0; i<D; ++i)
	{
		out << p[i];
		if(i < D-1)
			out << ';';
	}
	return out << ")";
}/*}}}*/
template <class T, int D> 
std::istream &operator>>(std::istream &in, Point<T,D> &pt)/*{{{*/
{
	if(in.peek() == '(')
		in.ignore();

	for(int i=0; i<D; ++i)
	{
		in >> pt[i];
		if(i<D-1 && (in.peek() == ';' || in.peek()==','))
			in.ignore();
	}

	if(!in.eof() && in.peek() == ')')
		in.ignore();
	return in;
}/*}}}*/

template <class T, int D>
size_t hash_value(const Point<T,D> &p)/*{{{*/
{
	return boost::hash_range(p.begin(), p.end());
}/*}}}*/

template <class V>
auto inv(V &&v)/*{{{*/
	-> typename std::enable_if<is_point<V>::value, V &&>::type
{
	for(auto it=v.begin(); it!=v.end(); ++it)
		*it = 1 / *it;
	return std::forward<V>(v);
}/*}}}*/

template <class V>
auto inv(V &v)/*{{{*/
	-> typename std::enable_if<is_point<V>::value, V>::type
{
	return inv(V(v));
}/*}}}*/

}} // namespace s3d::math

// $Id: point.hpp 3119 2010-09-09 13:04:31Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

