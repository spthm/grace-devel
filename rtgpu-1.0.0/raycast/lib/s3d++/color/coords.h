#ifndef S3D_COLOR_COORDS_H
#define S3D_COLOR_COORDS_H

#include "fwd.h"
#include "../math/coords.h"
#include "traits.h"
#include "../math/real.h"

namespace s3d { namespace color
{
using math::operator+;
using math::operator-;
using math::operator*;
using math::operator/;

using std::get;

template <class C, class S> class coords;

template <class C, class S>
class coords
	: public math::coords<C,S>
{
	typedef math::coords<C,S> coords_base;

	typedef C color_type;
public:
	static const size_t dim = traits::dim<S>::value;
	typedef typename traits::value_type<S>::type value_type;

	using coords_base::size;
	using coords_base::begin;
	using coords_base::end;

	coords() {}

	// We should be using constructor inheritance...
	template <class...ARGS>
	coords(ARGS &&...args) : coords_base(std::forward<ARGS>(args)...) {}

	color_type &operator+=(const color_type &that);
	color_type &operator-=(const color_type &that);
	color_type &operator*=(const color_type &that);
	color_type &operator/=(const color_type &that);

	template <class U, class = 
		typename std::enable_if<std::is_convertible<C,U>::value>::type>
	bool operator==(const U &that) const
	{
		auto it1 = begin(); auto it2 = that.begin();
		while(it1 != end())
			if(!equal(*it1++, *it2++))
				return false;

		return true;
	}

	template <class U>
	auto operator+=(const U &v)
		-> typename std::enable_if<
			std::is_convertible<U,value_type>::value &&
			traits::model<C>::value!=model::RADIANCE,
			color_type&>::type;

	template <class U>
	auto operator-=(const U &v)
		-> typename std::enable_if<
			std::is_convertible<U,value_type>::value &&
			traits::model<C>::value != model::RADIANCE,
			color_type&>::type;

	template <class U>
	auto operator*=(const U &v)
		-> typename std::enable_if<
			std::is_convertible<U,value_type>::value &&
			traits::model<C>::value!=model::RADIANCE,
			color_type&>::type;

	template <class U>
	auto operator/=(const U &v)
		-> typename std::enable_if<
			std::is_convertible<U,value_type>::value &&
			traits::model<C>::value!=model::RADIANCE,
			color_type&>::type;

	friend std::ostream &operator<<(std::ostream &out, const color_type &c)/*{{{*/
	{
		Vector<value_type,dim> v;
		for(size_t i=0; i<dim; ++i)
			v[i] = c[i];
		
		return out << v;
	}/*}}}*/
};

}}

#include "coords.hpp"

#endif
