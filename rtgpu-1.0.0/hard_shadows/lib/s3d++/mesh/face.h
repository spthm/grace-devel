/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License 
	version 3 as published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public 
	License along with S3D++. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef S3D_MESH_FACE_H
#define S3D_MESH_FACE_H

#include <vector>
#include <array>
#include <memory>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/range/reference.hpp>
#include "../util/gcc.h"
#include "../util/type_traits.h"
#include "fwd.h"

namespace s3d { namespace math
{

template <class T>
struct is_face
{
	static const bool value = false;

};

template <class M>
struct is_face<M&> : is_face<M>
{
};

template <class M>
struct is_face<const M> : is_face<M>
{
};

// face<T>  --------------------------------------------------------------

template <class T>
class face<T>
{
	typedef std::vector<T> vertices;
public:
	static const int dim = RUNTIME;

	typedef typename vertices::iterator iterator;
	typedef typename vertices::const_iterator const_iterator;

	typedef typename vertices::reverse_iterator reverse_iterator;
	typedef typename vertices::const_reverse_iterator const_reverse_iterator;

	typedef T vertex_type;
	typedef T value_type;

	face() {}
	face(const face &that) : m_vertices(that.m_vertices) {}
	face(face &&that) : m_vertices(std::move(that.m_vertices)) {}

	template <int N>
	explicit face(const face<T,N> &that);

	template <int N>
	explicit face(face<T,N> &&that);

	face(const std::initializer_list<vertex_type> &vertices);

	iterator begin() { return m_vertices.begin();; }
	iterator end() { return m_vertices.end(); }

	const_iterator begin() const { return m_vertices.begin(); }
	const_iterator end() const { return m_vertices.end(); }

	reverse_iterator rbegin() { return m_vertices.rbegin(); }
	reverse_iterator rend() { return m_vertices.rend(); }

	const_reverse_iterator rbegin() const { return m_vertices.rbegin(); }
	const_reverse_iterator rend() const { return m_vertices.rend(); }

	bool operator==(const face &that) const
		{ return m_vertices == that.m_vertices; }
	bool operator!=(const face &that) const
		{ return !operator==(that); }

	face &operator=(const face &that) = default;
	face &operator=(face &&that) 
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

// face<T,D>  --------------------------------------------------------------

template <class T, int D>
class face 
{
	typedef std::array<T,D> vertices;
public:
	static const int dim = D;

	typedef typename vertices::iterator iterator;
	typedef typename vertices::const_iterator const_iterator;

	typedef typename vertices::reverse_iterator reverse_iterator;
	typedef typename vertices::const_reverse_iterator const_reverse_iterator;

	typedef T vertex_type;
	typedef T value_type;

	face() {}
	face(const face &that) : m_vertices(that.m_vertices) {}
	face(face &&that) : m_vertices(std::move(that.m_vertices)) {}

	template <int N, class = typename std::enable_if<N==RUNTIME || N==D>::type>
	face(const face<T,N> &that);

	template <int N, class = typename std::enable_if<N==RUNTIME || N==D>::type>
	face(face<T,N> &&that);

	template <class... V, class 
		= typename std::enable_if<D==sizeof...(V)+1>::type>
	face(const vertex_type &v0, V &&...vertices);

	template <class... V, class 
		= typename std::enable_if<D==sizeof...(V)+1>::type>
	face(vertex_type &&v0, V &&...vertices);

	template <int N, class = typename std::enable_if<N==RUNTIME || N==D>::type>
	face &operator=(const face<T,N> &that);

	template <int N, class = typename std::enable_if<N==RUNTIME || N==D>::type>
	face &operator=(face<T,N> &&that);

	iterator begin() { return m_vertices.begin();; }
	iterator end() { return m_vertices.end(); }

	const_iterator begin() const { return m_vertices.begin(); }
	const_iterator end() const { return m_vertices.end(); }

	reverse_iterator rbegin() { return m_vertices.rbegin(); }
	reverse_iterator rend() { return m_vertices.rend(); }

	const_reverse_iterator rbegin() const { return m_vertices.rbegin(); }
	const_reverse_iterator rend() const { return m_vertices.rend(); }

	bool operator==(const face &that) const
		{ return m_vertices == that.m_vertices; }
	bool operator!=(const face &that) const
		{ return !operator==(that); }

	face &operator=(const face &that) = default;
	face &operator=(face &&that) 
		{ m_vertices = std::move(that.m_vertices); return *this; }

	size_t size() const { return m_vertices.size(); }
	bool empty() const { return D == 0; }

	vertex_type &front() { return m_vertices.front(); }
	const vertex_type &front() const { return m_vertices.front(); }

	vertex_type &back() { return m_vertices.back(); }
	const vertex_type &back() const { return m_vertices.back(); }

	vertex_type &operator[](size_t i);
	const vertex_type &operator[](size_t i) const;

private:
	vertices m_vertices;
};

template <class T, int D>
struct is_face<face<T,D>>
{
	static const bool value = true;
};

template <class F>
auto operator<<(std::ostream &out, const F &face)
	-> typename std::enable_if<is_face<F>::value, std::ostream &>::type;

template <class F, class A=real> 
auto centroid(const F &face, A *area=NULL)
	-> typename std::enable_if<is_face<F>::value, typename F::value_type>::type;

template <class A=real, class F> 
auto area(const F &face)
	-> typename std::enable_if<is_face<F>::value, A>::type;

template <class F> 
auto normal(const F &face)
	-> typename std::enable_if<is_face<F>::value,
		Vector<typename value_type<F,2>::type, F::value_type::dim>>::type;

template <template <class,int> class F, int D, class T> 
auto intersect(const F<Point<T,3>,D>, const Ray<T,3> &r)
	-> typename std::enable_if<is_face<F<Point<T,3>,D>>::value, T>::type;

template <class F>
auto bounds(const F &face)
	-> typename std::enable_if<is_face<F>::value,
		Box<typename value_type<F,2>::type, F::value_type::dim>>::type;

}} // namespace s3d::math

#include "face.hpp"

#endif

// $Id: face.h 3174 2010-10-28 19:11:43Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

