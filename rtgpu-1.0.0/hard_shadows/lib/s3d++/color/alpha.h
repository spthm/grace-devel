#ifndef S3D_COLOR_ALPHA_H
#define S3D_COLOR_ALPHA_H

#include "fwd.h"
#include "../math/real.h"
#include "traits.h"
#include "radiance.h"
#include "coords.h"
#include "cast.h"
#include "compositing.h"
#include "luminance.h"
#include "../mpl/vector_fwd.h"

namespace s3d 
{ 

namespace color 
{
	template <class C, channel_position P>
	class alpha_space;
}

// needs to be here :(
namespace math { namespace traits
{
	template <class C, color::channel_position P>
	struct dim<color::alpha_space<C,P>>
	{
		typedef mpl::vector_c<int,color::traits::dim<C>::value> type;
	};
}} // namespace math::traits

namespace color
{

namespace traits
{
	template <class C, channel_position P>
	struct space<alpha<C,P>> : space<C>
	{
	};

	template <class C, channel_position P>
	struct model<alpha<C,P>> : model<C>
	{
	};
}

namespace detail/*{{{*/
{
	template <class T>
	struct alpha_base
	{
		alpha_base() {}
		alpha_base(T _a) : a(_a) {}
		T a;
	};
}/*}}}*/

template <class C>
class alpha_space<C, LAST>/*{{{*/
	: public traits::space<C>::type
	, public detail::alpha_base<typename traits::value_type<C>::type>
{
public:
	typedef typename traits::value_type<C>::type value_type;
private:
	typedef detail::alpha_base<value_type> alpha_base;
	typedef typename traits::space<C>::type color_base;

public:
	static const size_t dim = traits::dim<C>::value + 1;

	alpha_space() {}

	alpha_space(const C &c, value_type a) 
		: color_base(c), alpha_base(a) {}

	typedef value_type *iterator;
	typedef const value_type *const_iterator;

	iterator begin() { return &*color_base::begin(); }
	const_iterator begin() const { return &*color_base::begin(); }

	iterator end() { return begin() + dim; }
	const_iterator end() const { return begin() + dim; }
};/*}}}*/

template <class C>
class alpha_space<C, FIRST>/*{{{*/
	: public detail::alpha_base<typename traits::value_type<C>::type>
	, public traits::space<C>::type
{
public:
	typedef typename traits::value_type<C>::type value_type;
private:
	typedef detail::alpha_base<value_type> alpha_base;
	typedef typename traits::space<C>::type color_base;
public:
	static const size_t dim = traits::dim<C>::value + 1;

	using alpha_base::a;

	alpha_space() {}
	alpha_space(const C &c, value_type a) 
		: alpha_base(a), color_base(c) {}

	typedef value_type *iterator;
	typedef const value_type *const_iterator;

	iterator begin() { return &a; }
	const_iterator begin() const { return &a; }

	iterator end() { return begin() + dim; }
	const_iterator end() const { return begin() + dim; }
};/*}}}*/

template <class C, channel_position P>
struct alpha
	: coords<alpha<C,P>,alpha_space<C, P>>
	, compositing_operators<alpha<C,P>>
{
	typedef coords<alpha,alpha_space<C, P>> coords_base;

	using coords_base::a;

	typedef typename coords_base::value_type value_type;

	template <class U>
	struct rebind { typedef alpha<typename s3d::rebind<C,U>::type,P> type; };

	alpha() {};

	template <class DUMMY=int, class = 
		 typename std::enable_if<sizeof(DUMMY) &&
				!std::is_same<C,radiance>::value>::type>
	explicit alpha(const radiance &c)
		: coords_base(C(c), math::map<value_type>(1))
	{
	}

	template <class DUMMY=int, class = 
		 typename std::enable_if<sizeof(DUMMY) &&
				!std::is_same<C,radiance>::value>::type>
	explicit alpha(const radiance_alpha &c)
		: coords_base(C(c), math::map<value_type>(c.a))
	{
	}

	template <class U>
	alpha(const alpha<U,P> &c) : coords_base(c.base(), c.a) {}

	alpha(const C &c, value_type a=math::map<value_type>(1))
		: coords_base(c, a) {}

	alpha(const alpha &c) = default;

	operator radiance_alpha() const
	{
		return radiance_alpha(*this);//{static_cast<const C &>(*this), a};
	}

	const C &base() const
	{
		return reinterpret_cast<const C &>(static_cast<const typename traits::space<C>::type &>(*this));
	}

	C &base()
	{
		return reinterpret_cast<C &>(static_cast<typename traits::space<C>::type &>(*this));
	}

	alpha &operator+=(const alpha &that) { base()+=that.base(); return *this;}
	alpha &operator-=(const alpha &that) { base()-=that.base(); return *this;}
	alpha &operator*=(const alpha &that) { base()*=that.base(); return *this;}
	alpha &operator/=(const alpha &that) { base()/=that.base(); return *this;}

	template <class U>
	auto operator+=(const U &v)
		-> typename std::enable_if<
			std::is_convertible<U,value_type>::value,alpha&>::type
		{ base() += v; return *this; }

	template <class U>
	auto operator-=(const U &v)
		-> typename std::enable_if<
			std::is_convertible<U,value_type>::value,alpha&>::type
		{ base() -= v; return *this; }

	template <class U>
	auto operator*=(const U &v)
		-> typename std::enable_if<
			std::is_convertible<U,value_type>::value,alpha&>::type
		{ base() *= v; return *this; }

	template <class U>
	auto operator/=(const U &v)
		-> typename std::enable_if<
			std::is_convertible<U,value_type>::value,alpha&>::type
		{ base() /= v; return *this; }

	friend alpha comp(const alpha &c1, real f1, const alpha &c2, real f2)
	{
		assert(greater_or_equal_than(f1,0) && less_or_equal_than(f1,1));
		assert(greater_or_equal_than(f2,0) && less_or_equal_than(f2,1));

		return alpha((c1*f1 + c2*f2).base(), c1.a*f1 + c2.a*f2);
	}
};

template <class C, channel_position P>
auto alpha_channel(const alpha<C,P> &c) -> real
{
	return unmap(c.a);
}

template <class CTO, channel_position PTO, class CFROM, channel_position PFROM>
struct cast_def<alpha<CFROM,PFROM>, alpha<CTO,PTO>>/*{{{*/
{
	typedef typename alpha<CFROM,PFROM>::value_type FROM;
	typedef typename alpha<CTO,PTO>::value_type TO;

	// larger integral -> smaller integral
	template <class DUMMY=int>
	static auto map(const alpha<CFROM,PFROM> &c)
		-> typename std::enable_if<sizeof(DUMMY) &&
			    std::is_integral<FROM>::value &&
			    std::is_integral<TO>::value &&
				(sizeof(FROM) >= sizeof(TO)), alpha<CTO,PTO>>::type
	{
		return alpha<CTO,PTO>(color_cast<CTO>(c.base()), 
						      c.a >> (sizeof(FROM)-sizeof(TO))*8);
	}

	// smaller integral -> larger integral
	template <class DUMMY=int>
	static auto map(const alpha<CFROM,PFROM> &c)
		-> typename std::enable_if<sizeof(DUMMY) &&
			    std::is_integral<FROM>::value &&
			    std::is_integral<TO>::value &&
				(sizeof(FROM) < sizeof(TO)), alpha<CTO,PTO>>::type
	{
		return alpha<CTO,PTO>(color_cast<CTO>(c.base()), 
						      c.a << (sizeof(TO)-sizeof(FROM))*8);
	}

	template <class DUMMY=int>
	static auto map(const alpha<CFROM,PFROM> &c)
		-> typename std::enable_if<sizeof(DUMMY) &&
			    (!std::is_integral<FROM>::value ||
			     !std::is_integral<TO>::value), alpha<CTO,PTO>>::type
	{
		return alpha<CTO,PTO>(color_cast<CTO>(c.base()), 
							  math::map<TO>(math::unmap(c.a)));
						      
	}
};/*}}}*/

template <class CTO, channel_position PTO, class CFROM>
struct cast_def<CFROM, alpha<CTO,PTO>>/*{{{*/
{
	static alpha<CTO,PTO> map(const CFROM &c)
	{
		return alpha<CTO,PTO>(color_cast<CTO>(c));
	}
};/*}}}*/

template <class CFROM, channel_position PFROM, class CTO>
struct cast_def<alpha<CFROM,PFROM>, CTO>/*{{{*/
{
	static CTO map(const alpha<CFROM,PFROM> &c)
	{
		return color_cast<CTO>(c.base());
	}
};/*}}}*/

template <class C, channel_position PFROM>
struct cast_def<alpha<C,PFROM>, C>/*{{{*/
{
	static C map(const alpha<C,PFROM> &c)
	{
		return color_cast<C>(c.base());
	}
};/*}}}*/

template <class C, channel_position PTO>
struct cast_def<C, alpha<C,PTO>>/*{{{*/
{
	static alpha<C,PTO> map(const C &c)
	{
		return alpha<C,PTO>(c);
	}
};/*}}}*/

// for disambiguation
template <class C, channel_position P>
struct cast_def<alpha<C,P>, alpha<C,P>>/*{{{*/
{
	static const alpha<C,P> &map(const alpha<C,P> &c) 
	{ 
		return c; 
	}
};/*}}}*/

} // namespace color

} // s3d

namespace std
{
	template <class C, s3d::color::channel_position P>
	struct make_signed<s3d::color::alpha<C,P>>
	{
		typedef s3d::color::alpha<typename s3d::make_signed<C>::type,P> type;
	};
	template <class C, s3d::color::channel_position P>
	struct make_unsigned<s3d::color::alpha<C,P>>
	{
		typedef s3d::color::alpha<typename s3d::make_unsigned<C>::type,P> type;
	};
}

#endif

