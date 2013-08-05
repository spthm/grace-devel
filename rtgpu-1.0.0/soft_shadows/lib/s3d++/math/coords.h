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

#ifndef S3D_MATH_COORD_H
#define S3D_MATH_COORD_H

#include <tuple>
#include <stdexcept>
#include <array>
#include <vector>
#include "../util/gcc.h"
#include "../util/type_traits.h"
#include "../util/multi_param.h"
#include "../util/memory_view.h"
#include "../mpl/at.h"
#include "../mpl/create_range.h"
#include "real.h"
#include "../color/traits.h"
#include "operators.h"
#include "traits.h"

namespace s3d { namespace math
{

namespace detail/*{{{*/
{
	template <class C, int D, int HEAD>
	void retrieve_size(const C &c, Size<std::size_t, D> &s, 
					   mpl::vector_c<int,HEAD>)
	{
		if(c.size() != 0)
			s[HEAD] = c.size();
	}

	template <class C, int D, int HEAD, int... TAIL>
	void retrieve_size(const C &c, Size<std::size_t, D> &s, 
					   mpl::vector_c<int,HEAD,TAIL...>)
	{
		if(c.size() != 0)
		{
			s[HEAD] = c.size();
			retrieve_size(c[0], s, mpl::vector_c<int,TAIL...>());
		}
	}

}/*}}}*/

template <class C, class S, coords_kind TYPE = traits::coords_kind<S>::value>
class coords;

template <size_t I, class T> struct element;

struct dimension_tag;

// Na falta de um template typedef...
template <int D>
struct dimension
{
	typedef multi_param<std::size_t, D, dimension_tag> type;
};

template <int D>
typename dimension<D>::type dim(const Size<std::size_t, D> &s)/*{{{*/
{
	typename dimension<D>::type d;
	std::copy(s.begin(), s.end(), d.begin());

	return d;
}/*}}}*/

template <class...ARGS>
typename dimension<sizeof...(ARGS)>::type dim(ARGS... args)/*{{{*/
{
	return {args...};
}/*}}}*/

template <class C, class S> 
class coords<C,S,HOMOGENEOUS> : public S, public operators/*{{{*/
{
public:
	typedef S base;
	typedef S space_type;
	typedef typename S::value_type value_type;
	typedef typename std::remove_reference<decltype(*std::declval<S>().begin())>::type element_type;
	typedef C derived_type;

	typedef typename traits::dim<space_type>::type dim_type;

	coords() {}

	coords(const coords &that) : space_type(that) {}
	coords(coords &&that) : space_type(std::move(that)) {}

	// Same space type (copy ctor)
	template<class T>
	coords(const coords<T,S> &that) /*{{{*/
		: base(that) {}/*}}}*/

	// Same container type (move ctor)
	template<class T>
	coords(coords<T,S> &&that) /*{{{*/
		: base(std::move(that)) {}/*}}}*/

	// This is static
	template<class T, class S2>
	coords(const coords<T,S2> &that, /*{{{*/
		typename std::enable_if<sizeof(T) && 
			is_static<coords>::value>::type* = NULL)
	{
		typedef typename traits::dim<S2>::type that_dim;

		static_assert(mpl::at<0,dim_type>::value==RUNTIME || 
					  mpl::at<0,that_dim>::value==RUNTIME || 
					  mpl::at<0,dim_type>::value == mpl::at<0,that_dim>::value, 
				"Mismatched dimensions");

		if(size() != that.size())
			throw std::runtime_error("Mismatched dimensions");

		std::copy(that.begin(), that.end(), begin());
	}/*}}}*/

	// This is runtime
	template<class T, class S2, class = typename std::enable_if<sizeof(T) && 
		is_runtime<coords>::value>::type>
	coords(const coords<T,S2> &that) /*{{{*/
		: base(dim(that.size()))
	{
		if(size() != that.size())
			throw std::runtime_error("Mismatched dimensions");

		std::copy(that.begin(), that.end(), begin());
	}/*}}}*/

	// These constructors will be gone by the time gcc supports C++0x's
	// constructor inheritance

	// This is a runtime view
	template <class U>
	coords(U *data, size_t size,/*{{{*/
		typename std::enable_if<sizeof(U) &&
			is_view<coords>::value &&
			is_runtime<coords>::value>::type* = NULL)
		: base(data, size)
	{
	}/*}}}*/

	// This is a static view
	template <class U, class = typename std::enable_if<sizeof(U) &&
		is_view<coords>::value &&
		is_static<coords>::value>::type>
	coords(U *data, size_t size)/*{{{*/
		: base(data)
	{
		assert((size_t)(mpl::at<0,dim_type>::value) == size);
	}/*}}}*/
	
	template <class U, class = typename std::enable_if<sizeof(U) &&
		is_view<coords>::value &&
		is_static<coords>::value>::type>
	coords(U *data) /*{{{*/
		: base(data)
	{
	}/*}}}*/

	template <class...ARGS, class = typename std::enable_if<
			is_runtime<coords>::value || 
			(sizeof...(ARGS)+1 == mpl::at<0,dim_type>::value)>::type>
	coords(element_type c1, ARGS... cn)/*{{{*/
		: base{{c1, static_cast<element_type>(cn)...}}
	{
	}/*}}}*/

	template <int D>
	coords(const multi_param<size_t,D,dimension_tag> &d,
		typename std::enable_if<D*0==0 &&
			is_runtime<coords>::value>::type* = NULL)
		: base(d) {}

	template <int D, class = typename std::enable_if<D*0==0 && 
		is_static<coords>::value>::type>
	coords(const multi_param<size_t,D,dimension_tag> &d)/*{{{*/
	{
//		if(d != size())
//			throw std::runtime_error("Mismatched dimension");
	}/*}}}*/

	// for color::alpha_space
	template <class U,
			 class = typename std::enable_if<
			 	 is_colorspace<U>::value &&
			 	 is_colorspace<C>::value>::type>
	//		 	 !std::is_convertible<C,element_type>::value>::type>
	coords(const U &c, value_type alpha)
		: base(c, alpha) {}

	template<class T>
	coords &operator=(coords<T,S> &&that) /*{{{*/
	{
		typedef typename traits::dim<S>::type that_dim;

		static_assert(mpl::at<0,dim_type>::value == mpl::at<0,dim_type>::value, 
					  "Mismatched dimensions");

		if(area(size())!=0 && size() != that.size())
			throw std::runtime_error("Mismatched dimensions");

		base::operator=(std::move(that));
		return *this;
	}/*}}}*/

	// this is runtime and isn't a view
	template<class T, class S2>
	auto operator=(const coords<T,S2> &that) /*{{{*/
		-> typename std::enable_if<sizeof(T) &&
				is_runtime<coords>::value &&
				!is_view<coords>::value, coords &>::type
	{
		typedef typename traits::dim<S2>::type that_dim;

		static_assert(mpl::at<0,dim_type>::value==RUNTIME || 
					  mpl::at<0,that_dim>::value ||
					  mpl::at<0,dim_type>::value == mpl::at<0,that_dim>::value,
					  "Mismatched dimensions");

		if(size() != that.size())
		{
			if(empty())
				base::operator=(space_type(dim(that.size())));
			else
				throw std::runtime_error("Mismatched dimensions");
		}

		std::copy(that.begin(), that.end(), begin());
		return *this;
	}/*}}}*/

	// this is runtime and is a view
	template<class T, class S2>
	auto operator=(const coords<T,S2> &that) /*{{{*/
		-> typename std::enable_if<sizeof(S2) &&
				is_runtime<coords>::value &&
				is_view<coords>::value, coords &>::type
	{
		typedef typename traits::dim<S2>::type that_dim;

		static_assert(mpl::at<0,dim_type>::value==RUNTIME || 
					  mpl::at<0,that_dim>::value ||
					  mpl::at<0,dim_type>::value == mpl::at<0,that_dim>::value,
					  "Mismatched dimensions");

		if(size() != that.size())
			throw std::runtime_error("Mismatched dimensions");

		std::copy(that.begin(), that.end(), begin());
		return *this;
	}/*}}}*/

	// this is static 
	template<class T, class S2>
	auto operator=(const coords<T,S2> &that) /*{{{*/
		-> typename std::enable_if<sizeof(T) &&
				is_static<coords>::value, coords &>::type
	{
		typedef typename traits::dim<T>::type that_dim;
		static_assert(mpl::at<0,dim_type>::value==RUNTIME || 
					  mpl::at<0,that_dim>::value ||
					  mpl::at<0,dim_type>::value == mpl::at<0,that_dim>::value,
					  "Mismatched dimensions");

		if(size() != that.size())
			throw std::runtime_error("Mismatched dimensions");

		std::copy(that.begin(), that.end(), begin());
		return *this;
	}/*}}}*/

	template <class DUMMY=int>
	auto size() const/*{{{*/
		-> typename std::enable_if<sizeof(DUMMY)!=0
				&& order<coords>::value<=1, std::size_t>::type
	{
		return std::distance(begin(), end());
	}/*}}}*/

	template <class DUMMY=int>
	auto size() const /*{{{*/
		-> typename std::enable_if<sizeof(DUMMY)!=0 
					&& (order<coords>::value>1),
				Size<std::size_t, order<coords>::value>>::type
	{
		Size<std::size_t, order<coords>::value> s(0);

		detail::retrieve_size(S::m_coords, s, 
		   typename mpl::create_range<int,0,order<coords>::value>::type());
		return s;
	}/*}}}*/

	bool empty() const/*{{{*/
	{
		return area(size()) == 0;
	}/*}}}*/

	typedef typename space_type::iterator iterator;
	typedef typename space_type::const_iterator const_iterator;
	typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

	using S::begin;
	using S::end;

	const_iterator cbegin() const { return begin(); }
	const_iterator cend() const { return end(); }

	reverse_iterator rbegin() { return reverse_iterator(end()); }
	reverse_iterator rend() { return reverse_iterator(begin()); }
	const_reverse_iterator rbegin() const { return reverse_iterator(end()); }
	const_reverse_iterator rend() const { return reverse_iterator(begin()); }

	const_reverse_iterator crbegin() const { return rbegin(); }
	const_reverse_iterator crend() const { return rend(); }

	template<class T, class S2, class =
		typename std::enable_if<
			std::is_convertible<typename coords<T,S2>::derived_type,
								derived_type>::value>::type>
	bool operator==(const coords<T,S2> &that) const /*{{{*/
	{
		typedef typename traits::dim<S>::type that_dim;
		static_assert(mpl::at<0,dim_type>::value == mpl::at<0,that_dim>::value,
					  "Mismatched dimensions");

		if(area(size())!=0 && size() != that.size())
			throw std::runtime_error("Mismatched dimensions");

		auto it1 = begin();
		auto it2 = that.begin();
		while(it1 != end())
			if(!equal(*it1++, *it2++))
				return false;

		return true;
	}/*}}}*/

	template<class T, class S2>
	bool operator!=(const coords<T,S2> &that) const /*{{{*/
	{
		return !operator==(that);
	}/*}}}*/

	template<class T, class S2, class =
		typename std::enable_if<
			std::is_convertible<typename coords<T,S2>::derived_type,
								derived_type>::value>::type>
	bool operator<(const coords<T,S2> &that) const /*{{{*/
	{
		typedef typename traits::dim<S>::type that_dim;
		static_assert(mpl::at<0,dim_type>::value == mpl::at<0,that_dim>::value,
					  "Mismatched dimensions");

		if(area(size())!=0 && size() != that.size())
			throw std::runtime_error("Mismatched dimensions");

		auto it1 = begin();
		auto it2 = that.begin();
		while(it1 != end())
			if(!less_than(*it1++, *it2++))
				return false;

		return true;
	}/*}}}*/

	element_type &operator[](int idx)
	{
		assert(idx >= 0);
		assert(idx < std::distance(begin(), end()));
		return *(begin() + idx);
	}
	const element_type &operator[](int idx) const 
	{
		assert(idx >= 0);
		assert(idx < std::distance(begin(), end()));
		return *std::next(begin(), idx);
	}

	template <int I>
	friend element_type &get(coords &c) { return c[I]; }

	template <int I>
	friend const element_type &get(const coords &c) { return c[I]; }


private:
#if HAS_SERIALIZATION
	friend class boost::serialization::access;

	template <class A>
	void serialize(A &ar, unsigned int version)
	{
		for(auto it=begin(); it!=end(); ++it)
			ar & *it;
	}
#endif
};/*}}}*/

template <class C, class S> 
class coords<C,S,HETEROGENEOUS> : public S, public operators/*{{{*/
{
public:
	typedef S space_type;
	typedef C derived_type;
	using S::dim;

	coords() {}

	template<class T, class S2>
	coords(const coords<T,S2> &that)
	{
		static_assert(dim == coords<T,S2>::dim, "Mismatch on the number of coordinates");
		assign(that, typename mpl::create_range<int,0,dim>::type());
	}

	template <class...ARGS> 
	coords(typename element<0,space_type>::type c1, ARGS... cn)
	{
		static_assert(sizeof...(ARGS)+1 == dim, 
					  "Invalid number of coordinates");
		assign(c1, cn...);
	}

	template<class T, class S2>
	bool operator==(const coords<T,S2> &that) const /*{{{*/
	{
		return equal(that, typename mpl::create_range<int,0,dim>::type());
	}/*}}}*/

	template<class T, class S2>
	bool operator!=(const coords<T,S2> &that) const /*{{{*/
	{
		return !operator==(that);
	}/*}}}*/
private:
	template <class T, class S2, int... II>
	void assign(const coords<T,S2> &c, const mpl::vector_c<int,II...> &)/*{{{*/
	{
		assign(get<II>(c)...);
	}

	void assign() {}
	template <class A, class...ARGS> void assign(A c1, ARGS... cn)
	{
		get<dim-sizeof...(ARGS)-1>(*this) = c1;
		assign(cn...);
	}/*}}}*/

	template <class T, class S2, int I, int... II>
	bool equal(const coords<T,S2> &c, const mpl::vector_c<int,I,II...> &) const/*{{{*/
	{
		if(get<I>(*this) != get<I>(c))
			return false;
		else
			return equal(c, mpl::vector_c<int,II...>());
	}

	template <class T, class S2>
	bool equal(const coords<T,S2> &c, const mpl::vector_c<int> &) const
	{
		return true;
	}/*}}}*/

	template <size_t I>
	friend typename element<I,space_type>::type &get(coords &c)
	{
		return element<I,space_type>::get(c);
	}
	template <size_t I>
	friend const typename element<I,space_type>::type &get(const coords &c)
	{
		return element<I,space_type>::get(c);
	}

#if HAS_SERIALIZATION
	friend class boost::serialization::access;

	template <class A>
	void serialize_element(A &ar, unsigned int version, mpl::vector_c<int>) {}

	template <class A, int I, int...II>
	void serialize_element(A &ar, unsigned int version, mpl::vector_c<int,I, II...>)
	{
		ar & get<I>(S::m_coords);
		serialize_element(ar, version, mpl::vector_c<int,II...>());
	}

	template <class A>
	void serialize(A &ar, unsigned int version)
	{
		serialize_element(ar, version, mpl::create_range<int,0,dim>::type());
	}
#endif
};/*}}}*/

namespace traits
{
	template <class S, class T, math::coords_kind TYPE>
	struct dim<coords<S,T,TYPE>>
	{
		typedef typename dim<T>::type type;
	};
}

}} // namespace s3d::math

#endif

// $Id: coords.h 3235 2010-12-08 21:17:49Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

