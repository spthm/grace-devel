#ifndef S3D_MESH_TRIANGLE_ADJACENCY_H
#define S3D_MESH_TRIANGLE_ADJACENCY_H

#include <vector>
#include <array>
#include "../math/r2/fwd.h"
#include "fwd.h"

namespace s3d { namespace math
{

template <class T>
class triangle_adjacency
{
	typedef std::array<T,6> vertices;
public:
	static const int dim = 6;

	typedef typename vertices::iterator iterator;
	typedef typename vertices::const_iterator const_iterator;

	typedef typename vertices::reverse_iterator reverse_iterator;
	typedef typename vertices::const_reverse_iterator const_reverse_iterator;

	typedef T vertex_type;
	typedef T value_type;

	triangle_adjacency() {}
	triangle_adjacency(const triangle_adjacency &that) 
		: m_vertices(that.m_vertices) {}
	triangle_adjacency(triangle_adjacency &&that) 
		: m_vertices(std::move(that.m_vertices)) {}

	template <class... V, class 
		= typename std::enable_if<sizeof...(V)+1 == dim>::type>
	triangle_adjacency(const vertex_type &v0, V &&...vertices);

	template <class... V, class 
		= typename std::enable_if<sizeof...(V)+1 == dim>::type>
	triangle_adjacency(vertex_type &&v0, V &&...vertices);

	iterator begin() { return m_vertices.begin();; }
	iterator end() { return m_vertices.end(); }

	const_iterator begin() const { return m_vertices.begin(); }
	const_iterator end() const { return m_vertices.end(); }

	reverse_iterator rbegin() { return m_vertices.rbegin(); }
	reverse_iterator rend() { return m_vertices.rend(); }

	const_reverse_iterator rbegin() const { return m_vertices.rbegin(); }
	const_reverse_iterator rend() const { return m_vertices.rend(); }

	bool operator==(const triangle_adjacency &that) const
		{ return m_vertices == that.m_vertices; }
	bool operator!=(const triangle_adjacency &that) const
		{ return !operator==(that); }

	triangle_adjacency &operator=(const triangle_adjacency &that) = default;
	triangle_adjacency &operator=(triangle_adjacency &&that) 
		{ m_vertices = std::move(that.m_vertices); return *this; }

	size_t size() const { return m_vertices.size(); }
	bool empty() const { return false; }

	vertex_type &front() { return m_vertices.front(); }
	const vertex_type &front() const { return m_vertices.front(); }

	vertex_type &back() { return m_vertices.back(); }
	const vertex_type &back() const { return m_vertices.back(); }

	vertex_type &operator[](size_t i);
	const vertex_type &operator[](size_t i) const;

private:
	vertices m_vertices;
};

// TODO: tá meio errado isso, mas é para poder ser elemento de um surface
template <class T>
struct is_face<triangle_adjacency<T>>
{
	static const bool value = true;
};

template <class T>
class triangle_adjacency_strip
{
	typedef std::vector<T> vertices;
public:
	typedef typename vertices::iterator iterator;
	typedef typename vertices::const_iterator const_iterator;

	typedef typename vertices::reverse_iterator reverse_iterator;
	typedef typename vertices::const_reverse_iterator const_reverse_iterator;

	typedef T vertex_type;
	typedef T value_type;

	triangle_adjacency_strip() {}
	triangle_adjacency_strip(const triangle_adjacency_strip &that) 
		: m_vertices(that.m_vertices) {}
	triangle_adjacency_strip(triangle_adjacency_strip &&that) 
		: m_vertices(std::move(that.m_vertices)) {}

	triangle_adjacency_strip(const std::initializer_list<vertex_type> &vtx);

	iterator begin() { return m_vertices.begin();; }
	iterator end() { return m_vertices.end(); }

	const_iterator begin() const { return m_vertices.begin(); }
	const_iterator end() const { return m_vertices.end(); }

	reverse_iterator rbegin() { return m_vertices.rbegin(); }
	reverse_iterator rend() { return m_vertices.rend(); }

	const_reverse_iterator rbegin() const { return m_vertices.rbegin(); }
	const_reverse_iterator rend() const { return m_vertices.rend(); }

	bool operator==(const triangle_adjacency_strip &that) const
		{ return m_vertices == that.m_vertices; }
	bool operator!=(const triangle_adjacency_strip &that) const
		{ return !operator==(that); }

	triangle_adjacency_strip &operator=(const triangle_adjacency_strip &that) 
		= default;
	triangle_adjacency_strip &operator=(triangle_adjacency_strip &&that) 
		{ m_vertices = std::move(that.m_vertices); return *this; }

	void reserve(size_t s) { m_vertices.reserve(s); }
	size_t size() const { return m_vertices.size(); }
	bool empty() const { return m_vertices.empty(); }
	void clear() { m_vertices.clear(); }
	void resize(size_t s) { m_vertices.resize(s); }

	vertex_type &front() { return m_vertices.front(); }
	const vertex_type &front() const { return m_vertices.front(); }

	vertex_type &back() { return m_vertices.back(); }
	const vertex_type &back() const { return m_vertices.back(); }

	template <class...ARGS>
	void push_back(ARGS &&...args);

	template <class V>
	auto value(int idx, V &&geom) const
#if GCC_VERSION >= 40500
		-> decltype(geom[std::declval<vertex_type>()])
#else
		-> typename std::remove_reference<V>::type::value_type 
#endif
	{
		assert((size_t)idx < m_vertices.size());
		assert((size_t)m_vertices[idx] < geom.size());
		return geom[m_vertices[idx]];
	}

	template <class IT, class IT2>
	void insert(IT itpos, IT2 itbeg, IT2 itend);

	void erase(iterator it)
	{
		m_vertices.erase(it);
	};

	vertex_type &operator[](size_t i);
	const vertex_type &operator[](size_t i) const;

private:
	vertices m_vertices;
};

// TODO: tá meio errado isso, mas é para poder ser elemento de um surface
template <class T>
struct is_face<triangle_adjacency_strip<T>>
{
	static const bool value = true;
};

auto make_triangle_adjacency_grid(std::vector<r2::param_coord> &coords,
								  const r2::box &a, size_t nu, size_t nv)
	-> surface<triangle_adjacency<int>>;

auto make_triangle_adjacency_strip_grid(std::vector<r2::param_coord> &coords,
										const r2::box &a, size_t nu, size_t nv)
	-> surface<triangle_adjacency_strip<int>>;


}} // namespace s3d::math

#include "triangle_adjacency.hpp"

#endif
