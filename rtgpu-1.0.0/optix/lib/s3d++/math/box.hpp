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

#include "hash.h"
#include "point.h"
#include "../mpl/vector.h"
#include "../mpl/create_range.h"

namespace s3d { namespace math
{

//{{{ Result types
template <class T, int D, class V>
struct result_add<Box<T,D>, vector_like<V>>/*{{{*/
{
	static_assert(detail::equaldim<Box<T,D>,V>::value, "Mismatched dimension");

	typedef typename result_add_dispatch<T,typename value_type<V>::type>::type
		subtype;

	typedef Box<subtype, D==RUNTIME?V::dim:D> type;
};/*}}}*/
template <class T, int D, class V>
struct result_sub<Box<T,D>, vector_like<V>>/*{{{*/
{
	static_assert(detail::equaldim<Box<T,D>,V>::value, "Mismatched dimension");

	typedef typename result_sub_dispatch<T,typename value_type<V>::type>::type
		subtype;

	typedef Box<subtype, D==RUNTIME?V::dim:D> type;
};/*}}}*/

template <class V, class U, int P>
struct result_add<vector_like<V>, Box<U,P>>/*{{{*/
	: result_add<Box<U,P>, vector_like<V>>
{
};/*}}}*/
template <class V, class U, int P>
struct result_sub<vector_like<V>, Box<U,P>>/*{{{*/
	: result_sub<Box<U,P>, vector_like<V>>
{
};/*}}}*/

template <class T, int D, class V>
struct result_add<Box<T,D>, size_like<V>>/*{{{*/
{
	static_assert(detail::equaldim<Box<T,D>,V>::value, "Mismatched dimension");

	typedef typename result_add_dispatch<T,typename value_type<V>::type>::type
		subtype;

	typedef Box<subtype, D==RUNTIME?V::dim:D> type;
};/*}}}*/
template <class T, int D, class V>
struct result_sub<Box<T,D>, size_like<V>>/*{{{*/
{
	static_assert(detail::equaldim<Box<T,D>,V>::value, "Mismatched dimension");

	typedef typename result_sub_dispatch<T,typename value_type<V>::type>::type
		subtype;

	typedef Box<subtype, D==RUNTIME?V::dim:D> type;
};/*}}}*/
template <class T, int D, class V>
struct result_mul<Box<T,D>, size_like<V>>/*{{{*/
{
	static_assert(detail::equaldim<Box<T,D>,V>::value, "Mismatched dimension");

	typedef typename result_mul_dispatch<T,typename value_type<V>::type>::type
		subtype;

	typedef Box<subtype, D==RUNTIME?V::dim:D> type;
};/*}}}*/
template <class T, int D, class V>
struct result_div<Box<T,D>, size_like<V>>/*{{{*/
{
	static_assert(detail::equaldim<Box<T,D>,V>::value, "Mismatched dimension");

	typedef typename result_div_dispatch<T,typename value_type<V>::type>::type
		subtype;

	typedef Box<subtype, D==RUNTIME?V::dim:D> type;
};/*}}}*/

template <class V, class U, int P>
struct result_add<size_like<V>, Box<U,P>>/*{{{*/
	: result_add<Box<U,P>, size_like<V>>
{
};/*}}}*/
template <class V, class U, int P>
struct result_sub<size_like<V>, Box<U,P>>/*{{{*/
	: result_sub<Box<U,P>, size_like<V>>
{
};/*}}}*/
template <class V, class U, int P>
struct result_mul<size_like<V>, Box<U,P>>/*{{{*/
	: result_mul<Box<U,P>, size_like<V>>
{
};/*}}}*/
template <class V, class U, int P>
struct result_div<size_like<V>, Box<U,P>>/*{{{*/
	: result_div<Box<U,P>, size_like<V>>
{
};/*}}}*/
//}}}

namespace detail
{
	template <template <class, int> class P, class T, int...II>
	auto retrieve_args(const T &args, mpl::vector_c<int,II...>)
		-> P<typename std::tuple_element<0,T>::type, sizeof...(II)>
	{
		typedef typename std::tuple_element<0,T>::type elem_type;

		return { static_cast<elem_type>(std::get<II>(args))... };
	}
}

template <class T, int D> template <class...ARGS, class>
Box<T,D>::Box(const T &a, const ARGS &...args)
	: coords_base(detail::retrieve_args<Point>(std::make_tuple(a, args...),
								typename mpl::create_range<int,0,D>::type()), 
				  detail::retrieve_args<Size>(std::make_tuple(a, args...),
				  				typename mpl::create_range<int,D,2*D>::type()))
{
}

template <class T, int D> 
Point<T,D> centroid(const Box<T,D> &b)/*{{{*/
{
	return b.origin + b.size/2;
}/*}}}*/
template <class T, class U, int D>
auto centered_box(const Point<T,D> &c, const Size<U,D> &s)/*{{{*/
	-> Box<typename value_type<decltype(c-s/2)>::type, D>
{
	return { c-s/2, s };
}/*}}}*/

template <class T, int D> 
Box<T,D> null_box()/*{{{*/
{
	Box<T,D> b;

	std::fill(b.origin.begin(), b.origin.end(), 0);
	std::fill(b.size.begin(), b.size.end(), 0);

	return b;
}/*}}}*/

template <class T, int D> 
Box<T,D> &Box<T,D>::operator *=(const T &b)/*{{{*/
{
	origin *= b;
	size *= b;
	return *this;
}/*}}}*/
template <class T, int D> 
Box<T,D> &Box<T,D>::operator /=(const T &b)/*{{{*/
{
	origin /= b;
	size /= b;
	return *this;
}/*}}}*/
template <class T, int D> 
Box<T,D> &Box<T,D>::operator -=(const T &b)/*{{{*/
{
	origin -= b;
	return *this;
}/*}}}*/
template <class T, int D> 
Box<T,D> &Box<T,D>::operator +=(const T &b)/*{{{*/
{
	origin += b;
	return *this;
}/*}}}*/

template <class T, int D> 
Box<T,D> &Box<T,D>::operator -=(const Vector<T,D> &v)/*{{{*/
{
	origin -= v;
	return *this;
}/*}}}*/
template <class T, int D> 
Box<T,D> &Box<T,D>::operator +=(const Vector<T,D> &v)/*{{{*/
{
	origin += v;
	return *this;
}/*}}}*/

template <class T, int D> 
Box<T,D> &Box<T,D>::operator *=(const Size<T,D> &s)/*{{{*/
{
	origin = (origin - math::origin<T,D>())*s + math::origin<T,D>();
	size *= s;
	return *this;
}/*}}}*/
template <class T, int D> 
Box<T,D> &Box<T,D>::operator /=(const Size<T,D> &s)/*{{{*/
{
	origin = (origin - math::origin<T,D>())/s + math::origin<T,D>();
	size /= s;
	return *this;
}/*}}}*/
template <class T, int D> 
Box<T,D> &Box<T,D>::operator -=(const Size<T,D> &s)/*{{{*/
{
	size -= s;
	return *this;
}/*}}}*/
template <class T, int D> 
Box<T,D> &Box<T,D>::operator +=(const Size<T,D> &s)/*{{{*/
{
	size += s;
	return *this;
}/*}}}*/

template <class T, int D> 
Box<T,D> &Box<T,D>::operator &=(const Box<T,D> &b)/*{{{*/
{
	if(*this == b || is_zero())
		;
	else if(!overlap(*this,b))
		*this = null_box<T,D>();
	else
	{
		assert(is_positive());
		assert(b.is_positive());

		Point<T,D> l = lower(upper(*this), upper(b)),
			u = upper(lower(*this), lower(b));

		origin = u;
		size = Size<T,D>(l-u);

		assert(is_positive());
	}

	return *this;
}/*}}}*/
template <class T, int D> 
Box<T,D> &Box<T,D>::operator |=(const Box<T,D> &b)/*{{{*/
{
	if(*this == b || b.is_zero())
		;
	else if(is_zero())
		*this = b;
	else
	{
		assert(is_positive());
		assert(b.is_positive());

		Point<T,D> l = lower(lower(*this), lower(b)),
			u = upper(upper(*this), upper(b));

		origin = l;
		size = Size<T,D>(u-l);

		assert(is_positive());
	}
	return *this;
}/*}}}*/

template <class T, int D, class U>
auto operator&(const Box<T,D> &a, const Box<U,D> &b)/*{{{*/
	-> typename result_mul<Box<T,D>,Box<U,D>>::type
{
	return typename result_mul<Box<T,D>,Box<U,D>>::type(b) &= a;
}/*}}}*/
template <class T, int D, class U>
auto operator|(const Box<T,D> &a, const Box<U,D> &b)/*{{{*/
	-> typename result_add<Box<T,D>,Box<U,D>>::type
{
	return typename result_add<Box<T,D>,Box<U,D>>::type(b) |= a;
}/*}}}*/

template <class T, int D> 
Box<T,D> &Box<T,D>::merge(const Point<T,D> &pt)/*{{{*/
{
	T eps = std::numeric_limits<T>::denorm_min();

	if(is_null())
	{
		origin = pt;
		size = Size<T,D>(eps);
	}
	else
	{
		// TODO: once gcc-4.5 is available, we will use explicit conversion
		// from size to vector. Meanwhile...
		Vector<T,D> vaux;
		std::copy(size.begin(), size.end(), vaux.begin());

		size = Size<T,D>(upper(vaux, pt-origin + eps));

		auto ito = origin.begin(),
			 its = size.begin();

		for(auto itpt = pt.begin(); itpt!=pt.end(); ++itpt, ++ito, ++its)
		{
			if(*itpt < *ito)
			{
				*its += *ito - *itpt;
				*ito = *itpt;
			}
		}
	}

	return *this;
}/*}}}*/
template <class T, class U, int D> 
bool overlap(const Box<T, D> &b1, const Box<U,D> &b2)/*{{{*/
{
	assert(b1.is_positive());
	assert(b2.is_positive());

	if(b1.is_zero() || b2.is_zero())
		return false;

	auto it2o = b2.origin.begin();
	auto it2s = b2.size.begin();
	auto it1s = b1.size.begin();

	for(auto it1o = b1.origin.begin(); it1o!=b1.origin.end(); 
		++it1o,++it2o,++it1s,++it2s)
	{
		if(*it2o >= *it1o + *it1s || *it1o >= *it2o + *it2s)
			return false;
	}

	return true;
}/*}}}*/

template <class U, class T, int D> 
Box<T,D> grow(const Box<T,D> &b, U d)/*{{{*/
{
	Size<T,D> newsize = b.size + 2*d;

	for(auto it=newsize.begin(); it!=newsize.end(); ++it)
		*it = max(std::numeric_limits<T>::denorm_min(), *it);

	return Box<T,D>(b.origin - d, newsize);
}/*}}}*/

template <class T> 
Box<T,2> transpose(const Box<T,2> &b)/*{{{*/
{
	return Box<T,2>(Point<T,2>(b.y, b.x), Size<T,2>(b.h, b.w));
}/*}}}*/

template <class T, int D> 
bool Box<T,D>::contains(const Point<T,D> &p) const/*{{{*/
{
	auto ito = origin.begin();
	auto its = size.begin();

	for(auto itp = p.begin(); itp != p.end(); ++itp, ++ito, ++its)
	{
		if(*itp < *ito || *itp >= *ito + *its)
			return false;
	}

	return true;
}/*}}}*/
template <class T, int D> 
bool Box<T,D>::contains(const Box<T,D> &bx) const/*{{{*/
{
	auto its = size.begin();
	auto itbo = bx.origin.begin();
	auto itbs = bx.size.begin();

	for(auto ito = origin.begin(); ito!=origin.end(); ++ito,++its,++itbo,++itbs)
	{
		if(*itbo < *ito || *itbo + *itbs > *ito + *its)
			return false;
	}

	return true;
}/*}}}*/

template <class T, int D> 
Point<T,D> Box<T,D>::constrain(const Point<T,D> &pt) const/*{{{*/
{
	assert(is_positive());

	Point<T,D> out;

	auto itout = out.begin();
	auto ito = origin.begin();
	auto its = size.begin();
	
	for(auto itpt = pt.begin(); itpt!=pt.end(); ++itpt, ++itout, ++ito, ++its)
		*itout = max(*ito, min(*ito + *its, *itpt));

	return out;
}/*}}}*/

template <class T, int D> 
typename std::enable_if<!std::is_integral<T>::value,Box<T,D>>::type normalize(const Box<T,D> &_r)/*{{{*/
{
	// Done this way to avoid spurious copies and comparisons
	bool ltw = less_than(_r.w, 0),
		 lth = less_than(_r.h, 0);

	if(ltw || lth)
	{
		Box<T,D> r;
		if(ltw)
		{
			r.x = _r.x + _r.w;
			r.w = -_r.w;
		}
		else
		{
			r.x = _r.x;
			r.w = _r.w;
		}
		if(lth)
		{
			r.y = _r.y + _r.h;
			r.h = -_r.h;
		}
		else
		{
			r.y = _r.y;
			r.h = _r.h;
		}
		assert(r.is_positive());
		return r;
	}
	else
	{
		assert(_r.is_positive());
		return _r;
	}
}/*}}}*/

template <class T, int D> 
typename std::enable_if<std::is_integral<T>::value,Box<T,D>>::type normalize(const Box<T,D> &_r)/*{{{*/
{
	// Done this way to avoid spurious copies and comparisons
	bool ltw = less_than(_r.w, 0),
		 lth = less_than(_r.h, 0);

	if(ltw || lth)
	{
		Box<T,D> r;
		if(ltw)
		{
			r.x = _r.x + _r.w;
			r.w = -_r.w+1;
		}
		else
		{
			r.x = _r.x;
			r.w = _r.w;
		}
		if(lth)
		{
			r.y = _r.y + _r.h;
			r.h = -_r.h+1;
		}
		else
		{
			r.y = _r.y;
			r.h = _r.h;
		}
		assert(r.is_positive());
		return r;
	}
	else
	{
		assert(_r.is_positive());
		return _r;
	}
}/*}}}*/

template <class T> 
real aspect(const Box<T,2> &b)/*{{{*/
{
	return real(b.w) / b.h;
}/*}}}*/
template <class T> 
real aspect(const Box<T,3> &b)/*{{{*/
{
	return real(b.w) / b.h;
}/*}}}*/

template <class T, int D> 
bool Box<T,D>::operator==(const Box &that) const/*{{{*/
{
	return origin == that.origin && size == that.size;
}/*}}}*/

template <class T, int D> 
const Point<T,D> &lower(const Box<T,D> &b)/*{{{*/
{
	return b.origin;
}/*}}}*/
template <class T, int D, class... ARGS> 
Point<T,D> lower(const Box<T,D> &b1, const ARGS &...args)/*{{{*/
{
	return lower(lower(b1), lower(args...));
}/*}}}*/

template <class T, int D> 
Point<T,D> upper(const Box<T,D> &b)/*{{{*/
{
	return b.origin + b.size;
}/*}}}*/
template <class T, int D, class... ARGS> 
Point<T,D> upper(const Box<T,D> &b1, const ARGS &...args)/*{{{*/
{
	return upper(upper(b1), upper(args...));
}/*}}}*/

template <class T, int D> 
bool Box<T,D>::is_null() const /*{{{*/
{ 
	return origin == Point<T,D>(0) && size == Size<T,D>(0);
}/*}}}*/
template <class T, int D> 
bool Box<T,D>::is_positive() const /*{{{*/
{ 
	return size.is_positive();
}/*}}}*/
template <class T, int D> 
bool Box<T,D>::is_negative() const /*{{{*/
{ 
	return size.is_negative();
}/*}}}*/
template <class T, int D> 
bool Box<T,D>::is_zero() const /*{{{*/
{ 
	return size.is_zero();
}/*}}}*/

template <class T, int D> 
T max_dim(const Box<T,D> &b)/*{{{*/
{
	return max_dim(b.size);
}/*}}}*/

template <class T, int D> 
T min_dim(const Box<T,D> &b)/*{{{*/
{
	return min_dim(b.size);
}/*}}}*/

template <class T, int D> 
Box<T,D> floor(const Box<T,D> &b)/*{{{*/
{
	return Box<T,D>(ceil(b.origin), floor(b.origin+b.size));
}/*}}}*/
template <class T, int D> 
Box<T,D> round(const Box<T,D> &b)/*{{{*/
{
	return Box<T,D>(round(b.origin), round(b.origin+b.size));
}/*}}}*/
template <class T, int D> 
Box<T,D> ceil(const Box<T,D> &b)/*{{{*/
{
	return Box<T,D>(floor(b.origin), ceil(b.origin+b.size));
}/*}}}*/

template <class T, int D> 
std::ostream &operator<<(std::ostream &out, const Box<T,D> &b)/*{{{*/
{
	return out << b.origin << "-" << b.size;
}/*}}}*/
template <class T, int D> 
std::istream &operator>>(std::istream &in, Box<T,D> &b)/*{{{*/
{
	in >> b.origin;
	if(in.peek() == '-' || in.peek()==',' || in.peek()==';')
		in.ignore();
	return in >> b.size;
}/*}}}*/

template <class T, int D>
size_t hash_value(const Box<T,D> &p)/*{{{*/
{
	size_t seed=0;
	boost::hash_combine(seed,p.origin);
	boost::hash_combine(seed,p.size);
	return seed;
}/*}}}*/

}} // namespace s3d::math

// $Id: box.hpp 3100 2010-09-02 22:03:54Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

