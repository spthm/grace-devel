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

#include <cassert>
#include <boost/foreach.hpp>
#include "../util/type_traits.h"
#include "../math/point.h"
#include "../math/box.h"

namespace s3d { namespace math
{

template <class T>
face<T>::face(const std::initializer_list<T> &vertices) /*{{{*/
	: m_vertices(vertices.begin(), vertices.end())
{
}/*}}}*/

template <class T, int D> template <class... TT, class>
face<T,D>::face(const T &v0,  TT&&...vv)/*{{{*/
	: m_vertices(vertices{{v0, std::forward<TT>(vv)...}})
{
}/*}}}*/

template <class T, int D> template <class... TT, class>
face<T,D>::face(T &&v0,  TT&&...vv)/*{{{*/
	: m_vertices(vertices{{std::move(v0), static_cast<T>(std::forward<TT>(vv))...}})
{
}/*}}}*/

template <class T> template <int N>
face<T>::face(const face<T,N> &that)/*{{{*/
{
	m_vertices.reserve(that.size());

	BOOST_FOREACH(auto &vtx, that)
		m_vertices.emplace_back(vtx);
}/*}}}*/

template <class T> template <int N>
face<T>::face(face<T,N> &&that)/*{{{*/
{
	m_vertices.reserve(that.size());

	BOOST_FOREACH(auto &vtx, that)
		m_vertices.emplace_back(std::move(vtx));
}/*}}}*/

template <class T, int D> template <int N, class>
face<T,D>::face(const face<T,N> &that)/*{{{*/
{
	if(size() != that.size())
		throw std::runtime_error("Dimension mismatch");

	auto it = begin();
	BOOST_FOREACH(auto &vtx, that)
		*it++ = vtx;
}/*}}}*/

template <class T, int D> template <int N, class>
face<T,D>::face(face<T,N> &&that)/*{{{*/
{
	if(size() != that.size())
		throw std::runtime_error("Dimension mismatch");

	auto it = begin();
	BOOST_FOREACH(auto &vtx, that)
		*it++ = std::move(vtx);
}/*}}}*/

template <class T> template <class IT, class IT2>
void face<T>::insert(IT itpos, IT2 itbeg, IT2 itend)/*{{{*/
{
	m_vertices.insert(itpos, itbeg, itend);
}/*}}}*/

template <class T> template <class...ARGS>
void face<T>::push_back(ARGS &&...args) /*{{{*/
{ 
	m_vertices.push_back(std::forward<ARGS>(args)...);
}/*}}}*/

template <class T, int D>
T &face<T,D>::operator[](size_t i) /*{{{*/
{ 
	assert(i < m_vertices.size());
	return m_vertices[i];
}/*}}}*/
template <class T, int D>
const T &face<T,D>::operator[](size_t i) const /*{{{*/
{ 
	assert(i < m_vertices.size());
	return m_vertices[i];
}/*}}}*/

template <class T>
T &face<T>::operator[](size_t i) /*{{{*/
{ 
	assert(i < m_vertices.size());
	return m_vertices[i];
}/*}}}*/
template <class T>
const T &face<T>::operator[](size_t i) const /*{{{*/
{ 
	assert(i < m_vertices.size());
	return m_vertices[i];
}/*}}}*/

template <class F, class A=real> 
auto centroid(const F &face, A *area=NULL)/*{{{*/
	-> typename std::enable_if<is_face<F>::value, typename F::value_type>::type
{
	// O exemplo de wireframe no scene.scn não fica legal quando usamos a
	// definição correta de centroide...
#if 0
	// ref: http://local.wasp.uwa.edu.au/~pbourke/geometry/polyarea/

	typename difference_type<T>::type c(0);

	typename T::value_type _2a(0);

	for(size_t i=0; i<f.size(); ++i)
	{
		auto t = f[i].x*f[i+1].y - f[i+1].x*f[i].y;
		_2a += t;

		auto dc = f[i] - origin<T::dim, typename T::value_type>();
		dc += f[i+1] - origin<T::dim, typename T::value_type>();

		c += dc*t;
	}

	if(area)
		*area = _2a / A(2);

	return c/(3*_2a) + origin<T::dim, typename T::value_type>();
#endif
#if 1
	typename difference_type<typename F::value_type>::type c(0);
	BOOST_FOREACH(auto &pt, face)
		c += pt - origin<typename value_type<F,2>::type, F::value_type::dim>();

	return c/face.size() + origin<typename value_type<F,2>::type,
									   F::value_type::dim>();
#endif
}/*}}}*/

template <class A=real, class F> 
auto area(const F &face)/*{{{*/
	-> typename std::enable_if<is_face<F>::value, A>::type
{
	// ref: http://local.wasp.uwa.edu.au/~pbourke/geometry/polyarea/

	A a = face.back().x * face.front().y - face.front().x * face.back().y;

	for(size_t i=0; i<face.size()-1; ++i)
		a += face[i].x * face[i+1].y - face[i+1].x * face[i].y;

	return a/2;
}/*}}}*/

template <class F> 
auto normal(const F &face)/*{{{*/
	-> typename std::enable_if<is_face<F>::value,
		Vector<typename value_type<F,2>::type, F::value_type::dim>>::type
{
	assert(face.size() >= 3);
	return cross(face[1] - face[0], face[2]-face[0]);
}/*}}}*/

template <template <class,int> class F, class T, int D> 
auto intersect(const F<Point<T,3>,D> &face, const Ray<T,3> &r) /*{{{*/
	-> typename std::enable_if<is_face<F<Point<T,3>,D>>::value, T>::type
{
	auto tri = make_triangle_traversal(face);

	T dist = std::numeric_limits<T>::max();

	do
	{
		dist = min(dist, intersect(tri(face), r));
	}
	while(++tri);

	if(dist == std::numeric_limits<T>::max())
		return -std::numeric_limits<T>::max();
	else
		return dist;
}/*}}}*/

template <class F>
auto bounds(const F &face)/*{{{*/
	-> typename std::enable_if<is_face<F>::value,
		Box<typename value_type<F,2>::type, F::value_type::dim>>::type
{
	auto bbox = math::null_box<typename value_type<F,2>::type, F::value_type::dim>();

	for(int i=0; i<face.size(); ++i)
		bbox |= face[i];

	return bbox;
}/*}}}*/

template <class F>
auto operator<<(std::ostream &out, const F &face)/*{{{*/
	-> typename std::enable_if<is_face<F>::value, std::ostream &>::type
{
	out << '{';
	if(!face.empty())
	{
		auto itv = face.begin();

		out << *itv++;

		for(;itv != face.end(); ++itv)
			out << ',' << *itv;
	}

	return out << '}';
}/*}}}*/

}} // namespace s3d::math

// $Id: face.hpp 3174 2010-10-28 19:11:43Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

