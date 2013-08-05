#ifndef S3D_MESH_EDGE_H
#define S3D_MESH_EDGE_H

#include "face.h"

namespace s3d { namespace math
{

template <class F>
class edge_view
{
	static_assert(is_face<F>::value, "Must be a face");
public:
	typedef typename F::value_type value_type;

	edge_view(const F &face, int v0, int v1)
		: m_view{v0,v1}, m_face(&face) {}

	const F &face() const { assert(m_face); return *m_face; }

	const value_type &operator[](int i) const 
	{ 
		return face()[index_at(i)];
	}

	const int &index_at(int i) const
	{
		assert(i>=0 && i<2);
		return m_view[i];
	}

	int &index_at(int i)
	{
		assert(i>=0 && i<2);
		return m_view[i];
	}

	template <class F2>
	edge_view<F2> rebind(const F2 &face) const
	{
		return edge_view<F2>(face, m_view[0], m_view[1]);
	}

private:
	int m_view[2];
	const F *m_face; // pointer to allow default assignment to be generated
};

template <class F>
class edge_traversal
{
	struct safe_bool { int a; };
public:
	edge_traversal(const F &face)
		: m_view(face, {face.size()-2, face.size()-1})
	{
	}

	edge_traversal &operator++()
	{
		assert(m_view.index_at(1) >= 1);
		--m_view.index_at(0);
		--m_view.index_at(1);
		return *this;
	}

	operator int safe_bool::*() const 
		{ return m_view.index_at(1) == 0 ? NULL : &safe_bool::a; }
	bool operator !() const { return !(*this == true); }

	operator const edge_view<F> &() const { return m_view; }

private:
	edge_view<F> m_view;
};

}} // namespace s3d::math

#endif
