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
#include <cmath>
#include <limits>
#include <functional>
#include <boost/functional/hash.hpp>
#include "../mpl/bool.h"
#include "../mpl/at.h"
#include "detail.h"
#include "traits.h"
#include "real.h"

namespace s3d { namespace math
{

//{{{ Operation result, equal vector types:
template <template<class,int> class V, class T, int N, class U, int P>
struct result_add<vector_like<V<T,N>>, vector_like<V<U,P>>>
{
	static_assert(N==RUNTIME || P==RUNTIME || N==P, "Mismatched dimension");

	typedef V<typename result_add_dispatch<T,U>::type, N==RUNTIME?P:N> type;
};

template <template<class,int> class V, class T, int N, class U, int P>
struct result_sub<vector_like<V<T,N>>, vector_like<V<U,P>>>
{
	static_assert(N==RUNTIME || P==RUNTIME || N==P, "Mismatched dimension");

	typedef V<typename result_sub_dispatch<T,U>::type, N==RUNTIME?P:N> type;
};

template <template<class,int> class V, class T, int N, class U, int P>
struct result_mul<vector_like<V<T,N>>, vector_like<V<U,P>>>
{
	static_assert(N==RUNTIME || P==RUNTIME || N==P, "Mismatched dimension");

	typedef V<typename result_mul_dispatch<T,U>::type, N==RUNTIME?P:N> type;
};

template <template<class,int> class V, class T, int N, class U, int P>
struct result_div<vector_like<V<T,N>>, vector_like<V<U,P>>>
{
	static_assert(N==RUNTIME || P==RUNTIME || N==P, "Mismatched dimension");

	typedef V<typename result_div_dispatch<T,U>::type, N==RUNTIME?P:N> type;
};/*}}}*/

//{{{ Operation result, different vector types: returns Vector<T,N>
template <class V1, class V2>
struct result_add<vector_like<V1>, vector_like<V2>>
{
	static_assert(V1::dim==RUNTIME || V2::dim==RUNTIME || V1::dim==V2::dim, 
				  "Mismatched dimension");

	typedef typename result_add_dispatch<typename value_type<V1>::type,
										 typename value_type<V2>::type>::type
				subtype;

	typedef Vector<subtype, V1::dim==RUNTIME?V2::dim:V1::dim> type;
};

template <class V1, class V2>
struct result_sub<vector_like<V1>, vector_like<V2>>
{
	static_assert(V1::dim==RUNTIME || V2::dim==RUNTIME || V1::dim==V2::dim, 
				  "Mismatched dimension");

	typedef typename result_sub_dispatch<typename value_type<V1>::type,
										 typename value_type<V2>::type>::type
				subtype;

	typedef Vector<subtype, V1::dim==RUNTIME?V2::dim:V1::dim> type;
};

template <class V1, class V2>
struct result_mul<vector_like<V1>, vector_like<V2>>
{
	static_assert(V1::dim==RUNTIME || V2::dim==RUNTIME || V1::dim==V2::dim, 
				  "Mismatched dimension");

	typedef typename result_mul_dispatch<typename value_type<V1>::type,
										 typename value_type<V2>::type>::type
				subtype;

	typedef Vector<subtype, V1::dim==RUNTIME?V2::dim:V1::dim> type;
};

template <class V1, class V2>
struct result_div<vector_like<V1>, vector_like<V2>>
{
	static_assert(V1::dim==RUNTIME || V2::dim==RUNTIME || V1::dim==V2::dim, 
				  "Mismatched dimension");

	typedef typename result_div_dispatch<typename value_type<V1>::type,
										 typename value_type<V2>::type>::type
				subtype;

	typedef Vector<subtype, V1::dim==RUNTIME?V2::dim:V1::dim> type;
};/*}}}*/

// Declared in size.h --------------------------------------
template <class T, int N> template <class U> 
Size<T,N>::Size(const Vector<U,N> &v)/*{{{*/
{
	std::copy(v.begin(), v.end(), begin());
}/*}}}*/

// Vector --------------------------------------------------

template <class T, int N> template <class DUMMY, class>
Vector<T,N>::Vector(const T &v)/*{{{*/
{
	std::fill(begin(), end(), v);
}/*}}}*/

template <class T, int N> 
Vector<T,N>::Vector(const dimension<1>::type &d, const T &v)/*{{{*/
	: coords_base(d)
{
	std::fill(begin(), end(), v);
}/*}}}*/

// assignment

template <class T, int N> 
template <template<class,int> class V, class U, int P, class>
auto Vector<T,N>::operator =(V<U,P> &&that) -> Vector &/*{{{*/
{
	static_assert(detail::equaldim<Vector, V<U,P>>::value, "Mismatched dimensions");

	// Assignment works if this is an empty vector (needed for swap)
	if(size()!=0 && size() != that.size())
		throw std::runtime_error("Mismatched dimensions");

	coords_base::operator=(std::move(that));
	return *this;
}/*}}}*/

template <class T, int N> 
template <template<class,int> class V, class U, int P, class>
auto Vector<T,N>::operator =(const V<U,P> &that) -> Vector &/*{{{*/
{
	static_assert(detail::equaldim<Vector, V<U,P>>::value, "Mismatched dimensions");

	// Assignment works if this is an empty vector (needed for swap)
	if(size()!=0 && size() != that.size())
		throw std::runtime_error("Mismatched dimensions");

	coords_base::operator=(that);
	return *this;
}/*}}}*/

// Negation

template <class V>
auto operator -(const V &v)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, V>::type
{
	V r(dim(v.size()));
	std::transform(v.begin(), v.end(), r.begin(), 
				   std::negate<typename V::value_type>());
	return r;
}/*}}}*/

// Equality

template <class V1, class V2>
auto operator==(const V1 &v1, const V2 &v2)/*{{{*/
	-> typename std::enable_if<is_vector<V1>::value && 
							   is_vector<V2>::value, bool>::type
{
	if(v1.size() != v2.size())
		return false;

	auto it1 = v1.begin(); auto it2 = v2.begin();
	while(it1 != v1.end())
		if(!equal(*it1++, *it2++))
			return false;
	return true;
}/*}}}*/

// Arithmetics Vector x Vector

template <class V1, class V2>
auto operator +=(V1 &&v1, const V2 &v2)/*{{{*/
	->  typename std::enable_if<is_vector<V1>::value && 
			(is_vector<V2>::value || is_size<V2>::value), V1 &&>::type
{
	static_assert(detail::equaldim<V1,V2>::value,"Mismatched dimensions");

	if(v1.size() != v2.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = v1.begin(); auto it2 = v2.begin();
	while(it1 != v1.end())
		*it1++ += *it2++;
	return std::forward<V1>(v1);
}/*}}}*/

template <class V1, class V2>
auto operator -=(V1 &&v1, const V2 &v2)/*{{{*/
	->  typename std::enable_if<is_vector<V1>::value && 
			(is_vector<V2>::value || is_size<V2>::value), V1 &&>::type
{
	static_assert(detail::equaldim<V1,V2>::value,"Mismatched dimensions");

	if(v1.size() != v2.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = v1.begin(); auto it2 = v2.begin();
	while(it1 != v1.end())
		*it1++ -= *it2++;
	return std::forward<V1>(v1);
}/*}}}*/

template <class V1, class V2>
auto operator *=(V1 &&v1, const V2 &v2)/*{{{*/
	->  typename std::enable_if<is_vector<V1>::value && 
			(is_vector<V2>::value || is_size<V2>::value), V1 &&>::type
{
	static_assert(detail::equaldim<V1,V2>::value,"Mismatched dimensions");

	if(v1.size() != v2.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = v1.begin(); auto it2 = v2.begin();
	while(it1 != v1.end())
		*it1++ *= *it2++;
	return std::forward<V1>(v1);
}/*}}}*/

template <class V1, class V2>
auto operator /=(V1 &&v1, const V2 &v2)/*{{{*/
	->  typename std::enable_if<is_vector<V1>::value && 
			(is_vector<V2>::value || is_size<V2>::value), V1&&>::type
{
	static_assert(detail::equaldim<V1,V2>::value,"Mismatched dimensions");

	if(v1.size() != v2.size())
		throw std::runtime_error("Mismatched dimensions");

	auto it1 = v1.begin(); auto it2 = v2.begin();
	while(it1 != v1.end())
		*it1++ /= *it2++;
	return std::forward<V1>(v1);
}/*}}}*/

// Arithmetics Vector x Real

template <class V>
auto operator +=(V &&v, typename value_type<V>::type d)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, V &&>::type
{
	for(auto it=v.begin(); it!=v.end(); ++it)
		*it += d;
	return std::forward<V>(v);
}/*}}}*/

template <class V>
auto operator -=(V &&v, typename value_type<V>::type d)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, V&&>::type
{
	for(auto it=v.begin(); it!=v.end(); ++it)
		*it -= d;
	return std::forward<V>(v);
}/*}}}*/

template <class V>
auto operator *=(V &&v, typename value_type<V>::type d)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, V&&>::type
{
	for(auto it=v.begin(); it!=v.end(); ++it)
		*it *= d;
	return std::forward<V>(v);
}/*}}}*/

template <class V>
auto operator /=(V &&v, typename value_type<V>::type d)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, V&&>::type
{
	for(auto it=v.begin(); it!=v.end(); ++it)
		*it /= d;
	return std::forward<V>(v);
}/*}}}*/

// Input / output

namespace detail_math
{
	template <class T>
	void output_value(std::ostream &out, const T &v, mpl::bool_<false>)
	{
		out << v;
	}
	template <class T>
	void output_value(std::ostream &out, const T &v, mpl::bool_<true>)
	{
		out << (int)(unsigned char)v;
	}
}

template <class V> 
auto operator<<(std::ostream &out, const V &v)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value,std::ostream &>::type

{
	typedef typename V::value_type value_type;

	out << "["; 
	for(std::size_t i=0; i<v.size(); ++i)
	{
		detail_math::output_value(out,v[i],
			mpl::bool_<std::is_integral<value_type>::value &&
			           sizeof(value_type)==1>());
		if(i < v.size()-1)
			out << ',';
	}
	return out << "]";
}/*}}}*/
template <class T, int N> 
std::istream &operator>>(std::istream &in, Vector<T,N> &v)/*{{{*/
{
	if(in.peek() == '[')
		in.ignore();

	for(std::size_t i=0; i<v.size(); ++i)
	{
		in >> v[i];
		if(i < v.size()-1 && (in.peek() == ';' || in.peek()==','))
			in.ignore();
	}

	if(!in.eof() && in.peek() == ']')
		in.ignore();
	return in;
}/*}}}*/

// Cross/dot/perp_dot/... products

template <class T, class U,
		  template<class,int> class V1, template<class,int> class V2,
	class = typename std::enable_if<is_vector<V1<T,2>>::value && 
									is_vector<V2<U,2>>::value>::type>
auto cross(const V1<T,2> &a, const V2<U,2> &b)/*{{{*/
	-> decltype(a.x*b.y - a.y*b.x)
{
	return a.x*b.y - a.y*b.x;
}/*}}}*/

template <class T, class U, 
		  template<class,int> class V1, template<class,int> class V2,
	class = typename std::enable_if<is_vector<V1<T,2>>::value && 
									is_vector<V2<U,2>>::value>::type>
auto cross(const V1<T,3> &a, const V2<U,3> &b)/*{{{*/
	-> Vector<decltype(a.y*b.z - a.z*b.y),3>
{
	return { a.y*b.z - a.z*b.y, 
			 a.z*b.x - a.x*b.z, 
			 a.x*b.y - a.y*b.x};
}/*}}}*/

template <class T, class U, int N, int P,
		  template<class,int> class V1, template<class,int> class V2,
	class = typename std::enable_if<is_vector<V1<T,N>>::value && 
								    is_vector<V2<U,P>>::value && 
							   (N==RUNTIME || P==RUNTIME)>::type>
auto cross(const V1<T,N> &a, const V2<U,P> &b)/*{{{*/
	-> Vector<decltype(a.x*b.y-a.y*b.x), N==RUNTIME?P:N>
{
	if(a.size() != b.size())
		throw std::runtime_error("Mismatched dimensions");

	// TODO: too much copying around... try to fix this
	switch(a.size())
	{
	case 2:
		return cross(V1<T,2>(a), V2<U,2>(b));
	case 3:
		return cross(V1<T,3>(a), V2<U,3>(b));
	default:
		throw std::runtime_error("Invalid dimension");
	}
}/*}}}*/

template <class T, class U,
	class = typename std::enable_if<
	(is_vector<T>::value || is_point<T>::value) && 
	(is_vector<U>::value || is_point<U>::value)>::type>
auto dot(const T &v1, const U &v2)/*{{{*/
	-> typename std::common_type<typename value_type<T>::type,
	                    typename value_type<U>::type>::type
{
	static_assert(detail::equaldim<T,U>::value, "Mismatched dimensions");

	if(v1.size() != v2.size())
		throw std::runtime_error("Mismatched dimensions");

	typedef typename std::common_type
	<
		typename value_type<T>::type,
	    typename value_type<U>::type
	>::type ret_type;

	ret_type d = 0;

	auto it1 = v1.begin(); auto it2 = v2.begin();

	while(it1 != v1.end())
		d += *it1++ * *it2++;
	return d;
}/*}}}*/

template <class V1, class V2,
	class = typename std::enable_if<is_vector<V1>::value && 
								    is_vector<V2>::value>::type>
auto perp_dot(const V1 &v1, const V2 &v2)/*{{{*/
	-> decltype(norm(cross(v1,v2)))
{
	return norm(cross(v1,v2)); // == |v1||v2|sin(theta)
}/*}}}*/

template <class A=real, class T, template<class,int> class V,
	class = typename std::enable_if<is_vector<V<T,2>>::value>::type>
A angle(const V<T,2> &v)/*{{{*/
{
	// Retorna de -Pi a Pi radianos
	return atan2(v.y, v.x);
}/*}}}*/

#if GCC_VERSION < 40500
namespace detail
{
	template <class V, class N>
	struct ret_proj
	{
		typedef decltype(dot(std::declval<V>(), std::declval<N>())*std::declval<N>()) type;
	};
};
#endif

template <class V1, class V2,
	class = typename std::enable_if<is_vector<V1>::value && 
								    is_vector<V2>::value>::type>
auto proj(const V1 &v, const V2 &n)/*{{{*/
#if GCC_VERSION >= 40500
	-> decltype(dot(v,n)*n)
#else
	-> typename detail::ret_proj<decltype(v), decltype(n)>::type
#endif
{
	return dot(v,n)*n;
}/*}}}*/

template <class V1, class V2,
	class = typename std::enable_if<is_vector<V1>::value && 
								    is_vector<V2>::value>::type>
auto ortho(const V1 &v, const V2 &n)/*{{{*/
	-> decltype(v-proj(v,n))
{
	return v-proj(v,n);
}/*}}}*/

template <int D1, int D2, int D3, class T> 
auto plane_normal(const Point<T,D1> &p1, const Point<T,D2>&p2,/*{{{*/
					     const Point<T,D3> &p3)
	-> decltype(cross(p1-p2,p3-p2))
{
	static_assert(detail::equaldim<Point<T,D1>, Point<T,D2>>::value &&
				  detail::equaldim<Point<T,D2>, Point<T,D3>>::value,
				  "Mismatched dimensions");

	if(p1.size() != p2.size() && p2.size() != p3.size())
		throw std::runtime_error("Mismatched dimensions");

	return cross(p1-p2, p3-p2);
}/*}}}*/

template <class A=real, class T, class U, class V,
		  template<class,int> class V1, template<class,int> class V2>
A angle(const V1<T,3> &a, const V2<U,3> &b, const Vector<V,3> &n)/*{{{*/
{
	auto pa = unit(ortho(a, n)),
		 pb = unit(ortho(b, n));

	auto c = cross(pa, pb);

	A ang = angle(pa, pb);
	if(dot(c, n) <= std::numeric_limits<A>::epsilon()*2)
	{
		ang = 2*M_PI-ang;
		if(ang > M_PI)
			ang -= 2*M_PI;
	}
	return ang;
}/*}}}*/

// Minimum angle between two vectors (range: 0 to pi radians)
template <class V1, class V2>
auto angle(const V1 &v1, const V2 &v2)/*{{{*/
	-> typename std::enable_if<is_vector<V1>::value && is_vector<V2>::value,
		decltype(acos(dot(v1,v2)))>::type
{
	return acos(dot(unit(v1), unit(v2)));
}/*}}}*/

template <class V> 
auto round(const V &p)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, V>::type
{
	V out(dim(p.size()));

	auto it1 = out.begin(); auto it2 = p.begin();

	while(it1 != out.end())
		*it1++ = round(*it2++);
	return out;
}/*}}}*/

template <class V> 
auto lower(const V &p)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, const V &>::type
{
	return p;
}/*}}}*/

template <template<class,int> class V, class T, int N> 
auto floor(const V<T,N> &v)/*{{{*/
	-> typename std::enable_if<is_vector<V<T,N>>::value, Vector<T,N>>::type
{
	Vector<T,N> out(v.size());

	auto it1 = out.begin(); auto it2 = v.begin();
	while(it1 != out.end())
		*it1++ = floor(*it2++);
	return out;
}/*}}}*/

template <template<class,int> class V, class T, int N> 
auto ceil(const V<T,N> &v)/*{{{*/
	-> typename std::enable_if<is_vector<V<T,N>>::value, Vector<T,N>>::type
{
	Vector<T,N> out(v.size());

	auto it1 = out.begin(); auto it2 = v.begin();
	while(it1 != out.end())
		*it1++ = ceil(*it2++);

	return out;
}/*}}}*/

template <class V>
auto inv(V &&v)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, V &&>::type
{
	for(auto it=v.begin(); it!=v.end(); ++it)
		const_cast<typename std::remove_const<typename value_type<V>::type>::type &>(*it) = 1 / *it;
	return std::forward<V>(v);
}/*}}}*/

template <class V>
auto inv(V &v)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, V>::type
{
	return inv(V(v));
}/*}}}*/

template <class V>
auto inv(const V &v)/*{{{*/
	-> typename std::enable_if<is_vector<V>::value, V>::type
{
	return inv(V(v));
}/*}}}*/

template <class T, int N, class... ARGS> 
Vector<T,N> lower(const Vector<T,N> &p1, const ARGS &...args)/*{{{*/
{
	static_assert(N!=RUNTIME, "Function doesn't work with dynamic vectors");

	Vector<T,N> p2 = lower(args...), l;

	auto it1 = p1.begin();
	auto it2 = p2.begin();
	auto itr = l.begin();

	while(it1 != p1.end())
		*itr++ = min(*it1++, *it2++);

	return l;
}/*}}}*/

template <class T, int N> 
const Vector<T,N> &upper(const Vector<T,N> &p)/*{{{*/
{
	return p;
}/*}}}*/

template <class T, int N, class... ARGS> 
Vector<T,N> upper(const Vector<T,N> &p1, const ARGS &...args)/*{{{*/
{
	static_assert(N!=RUNTIME, "Function doesn't work with dynamic vectors");

	Vector<T,N> p2 = upper(args...), u;

	auto it1 = p1.begin();
	auto it2 = p2.begin();
	auto itr = u.begin();

	while(it1 != p1.end())
		*itr++ = max(*it1++, *it2++);

	return u;
}/*}}}*/

template <template <class,int> class V, class T, class U, int M, int N> 
auto concat(const V<T,M> &v1, const V<U,N> &v2)/*{{{*/
	-> typename std::enable_if<M!=RUNTIME && N!=RUNTIME,
			V<typename std::common_type<T,U>::type, M+N>>::type
		
{
	V<typename std::common_type<T,U>::type, M+N> r;
		
	std::copy(v1.begin(), v1.end(), r.begin());
	std::copy(v2.begin(), v2.end(), r.begin()+v1.size());
	return r;
}/*}}}*/

template <template <class,int=RUNTIME> class V, class T, class U, int M, int N> 
auto concat(const V<T,M> &v1, const V<U,N> &v2)/*{{{*/
	-> typename std::enable_if<M==RUNTIME || N==RUNTIME,
			V<typename std::common_type<T,U>::type>>::type
{
	V<typename std::common_type<T,U>::type> r(dim(v1.size()+v2.size()));

	std::copy(v1.begin(), v1.end(), r.begin());
	std::copy(v2.begin(), v2.end(), r.begin()+v1.size());

	return r;
}/*}}}*/

template <template <class,int> class V, class T, int N, class... ARGS> 
auto augment(const V<T,N> &p, const ARGS &...c)/*{{{*/
	-> V<typename std::common_type<T,ARGS...>::type, N+sizeof...(ARGS)>
{
	return concat(p, V<typename std::common_type<ARGS...>::type,
					 sizeof...(ARGS)>(c...));
}/*}}}*/

template <int P, class T, int N> 
Vector<T,P> reduce(const Vector<T,N> &p)/*{{{*/
{
	static_assert(P < N, "Vector reduction doesn't reduce");

	Vector<T,P> r;
	std::copy(p.begin(), p.end()+P, r.begin());

	return r;
}/*}}}*/

template <class T, int N> 
Vector<T,N> reduce(const Vector<T> &p)/*{{{*/
{
	if(N < p.size())
		throw std::runtime_error("Vector reduction doesn't reduce");

	Vector<T,N> r;
	std::copy(p.begin(), p.begin()+N, r.begin());

	return r;
}/*}}}*/

template <class T, int N>
size_t hash_value(const Vector<T,N> &p)/*{{{*/
{
	return boost::hash_range(p.begin(), p.end());
}/*}}}*/

}} // namespace s3d::math


namespace std
{
	template <class T, int N> 
	struct hash<s3d::math::Vector<T,N>>/*{{{*/
	{
		size_t operator()(s3d::math::Vector<T,N> p) const
		{
			return hash_value(p);
		}
	};/*}}}*/
}

// $Id: vector.hpp 3115 2010-09-06 17:41:29Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

