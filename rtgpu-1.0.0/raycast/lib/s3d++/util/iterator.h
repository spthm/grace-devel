#ifndef S3D_UTIL_ITERATOR_H
#define S3D_UTIL_ITERATOR_H

#include <boost/iterator/iterator_adaptor.hpp>
#include "gcc.h"

#if GCC_VERSION < 40400
namespace std
{
	template<typename _InputIterator>
	inline _InputIterator
	next(_InputIterator __x, typename
		 iterator_traits<_InputIterator>::difference_type __n = 1)
	{
		std::advance(__x, __n);
		return __x;
	}

	template<typename _BidirectionalIterator>
	inline _BidirectionalIterator
	prev(_BidirectionalIterator __x, typename
		 iterator_traits<_BidirectionalIterator>::difference_type __n = 1)
	{
		std::advance(__x, -__n);
		return __x;
	}
}
#endif

namespace s3d
{

template <class T, class IT>
class static_cast_iterator
	: public boost::iterator_adaptor<static_cast_iterator<T,IT>, IT, T>
{
private:
	struct enabler {};
public:
	static_cast_iterator() 
		: static_cast_iterator::iterator_adaptor_(IT()) {}

	static_cast_iterator(IT it)
		: static_cast_iterator::iterator_adaptor_(it) {}

	template <class U, class IT2>
	static_cast_iterator(const static_cast_iterator<U,IT2> &that,
		typename std::enable_if<std::is_convertible<IT2,IT>::value,
								enabler>::type = enabler())
		: static_cast_iterator::iterator_adaptor_(that.base()) {}

	operator IT() const { return this->base(); }

	friend static_cast_iterator next(static_cast_iterator it) { return ++it; }
	friend static_cast_iterator prev(static_cast_iterator it) { return --it; }

private:
	friend class boost::iterator_core_access;
	T &dereference() const
	{
		return static_cast<T &>(*static_cast_iterator::iterator_adaptor_::base_reference());
	}
};

}

#endif
