#ifndef S3D_COLOR_LUMINANCE_H
#define S3D_COLOR_LUMINANCE_H

#include <array>
#include "fwd.h"

namespace s3d { namespace color
{

template <class T> 
struct luminance_space/*{{{*/
{
	static const int dim = 1;

	typedef T value_type;
	typedef std::array<typename std::remove_const<T>::type,dim> 
		container_type;

	luminance_space() {}

	template <class U>
	luminance_space(const luminance_space<U> &that) 
		: m_coords(that.m_coords) {}

	luminance_space(T _y)
		: m_coords((container_type){{_y}}) {}

	template <class U>
	luminance_space &operator=(const luminance_space<U> &that) 
	{
		m_coords = that.m_coords;
		return *this;
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

	union
	{
		T y;
		container_type m_coords;
	};
};/*}}}*/

}} // namespace s3d::color

#endif
