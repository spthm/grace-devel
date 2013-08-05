#ifndef S3D_MESH_TRIANGLE_VIEW_H
#define S3D_MESH_TRIANGLE_VIEW_H

#include <algorithm>
#include <cassert>
#include "fwd.h"

namespace s3d { namespace math
{

template <class F>
class triangle_view
{
	static_assert(is_face<F>::value, "Must be a face/face_view");
public:
	typedef typename F::value_type value_type;

	triangle_view(const F &face, int v0, int v1, int v2)
		: m_view{v0,v1,v2}, m_face(&face) {}

	const F &face() const { assert(m_face); return *m_face; }

	const value_type &operator[](int i) const 
	{ 
		return face()[index_at(i)];
	}

	const int &index_at(int i) const
	{
		assert(i>=0 && i<3);
		return m_view[i];
	}

	int &index_at(int i)
	{
		assert(i>=0 && i<3);
		return m_view[i];
	}

	template <class F2>
	triangle_view<F2> rebind(const F2 &face) const
	{
		return triangle_view<F2>(face, m_view);
	}

private:
	int m_view[3];
	const F *m_face; // pointer to allow default assignment to be generated
};

template <class F>
struct is_face<triangle_view<F>>
{
	static const bool value = true;
};

template <class F>
class triangle_traversal
{
	struct safe_bool { int a; };
public:
	triangle_traversal(const F &face)
		: m_view(face, 0, face.size()-2, face.size()-1) {}

	triangle_traversal &operator++()
	{
		assert(m_view.index_at(1) >= 1);
		--m_view.index_at(1);
		--m_view.index_at(2);
		return *this;
	}

	operator int safe_bool::*() const 
		{ return m_view.index_at(1) == 0 ? NULL : &safe_bool::a; }
	bool operator !() const { return !(*this == true); }

	const triangle_view<F> &get() const { return m_view; }
	triangle_view<F> &get() { return m_view; }

public:
	triangle_view<F> m_view;
};

template <class F>
triangle_traversal<F> make_triangle_traversal(const F &face)
{
	return triangle_traversal<F>(face);
}

}} // namespace s3d::math

#include "triangle_view.hpp"

#endif
