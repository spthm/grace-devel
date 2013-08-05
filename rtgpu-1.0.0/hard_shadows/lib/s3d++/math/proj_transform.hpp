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

#include "box.h"

namespace s3d { namespace math
{


template <class T, int D> 
ProjTransform<T,D>::ProjTransform(const ProjTransform &xf)/*{{{*/
{
	*this = xf;
}/*}}}*/

template <class T, int D> 
ProjTransform<T,D>::ProjTransform(ProjTransform &&xf)/*{{{*/
{
	*this = std::move(xf);
}/*}}}*/

template <class T, int D> template <class XF, class>
ProjTransform<T,D>::ProjTransform(const XF &xf)/*{{{*/
{
	*this = xf;
}/*}}}*/

template <class T, int D> template <class XF, class>
ProjTransform<T,D>::ProjTransform(XF &&xf)/*{{{*/
{
	*this = std::move(xf);
}/*}}}*/

template <class T, int D>
ProjTransform<T,D>::ProjTransform(const std::initializer_list<Vector<T,D+1>> &rows)/*{{{*/
	: AffineTransform<T,D>(typename base::init_base_xform_tag(), rows)
{
}/*}}}*/

template <class T, int D> template <int M, int N>
ProjTransform<T,D>::ProjTransform(const Matrix<T,M,N> &m)/*{{{*/
{
	static_assert(M>=D+1 && N>=D+1, "Matrix dimension must be >= D+1 x D+1");

	Matrix<T,D+1,D+1> &td = this->direct();

	for(int i=0; i<D+1; ++i)
		std::copy(m[i].begin(), m[i].begin()+D+1, td[i].begin());
}/*}}}*/

template <class T, int D> template <class XF, class>
ProjTransform<T,D> &ProjTransform<T,D>::operator=(const XF &xf)/*{{{*/
{
	Transform<T,D>::operator=(xf);
	return *this;
}/*}}}*/
template <class T, int D> template <class XF, class>
ProjTransform<T,D> &ProjTransform<T,D>::operator=(XF &&xf)/*{{{*/
{
	Transform<T,D>::operator=(std::move(xf));
	return *this;
}/*}}}*/

template <class T, int D>
ProjTransform<T,D> &ProjTransform<T,D>::operator=(const ProjTransform &xf)/*{{{*/
{
	Transform<T,D>::operator=(xf);
	return *this;
}/*}}}*/

template <class T, int D>
ProjTransform<T,D> &ProjTransform<T,D>::operator=(ProjTransform &&xf)/*{{{*/
{
	Transform<T,D>::operator=(std::move(xf));
	return *this;
}/*}}}*/

template <class T, int D> 
Vector<T,D+1> &ProjTransform<T,D>::operator[](int i)/*{{{*/
{
	this->modified();
	return this->direct()[i];
}/*}}}*/


template <class T, int D> template <class P> 
auto ProjTransform<T,D>::do_multiply_impl(const P &p) const/*{{{*/
	-> typename std::enable_if<is_point<P>::value, P>::type
{
	// [a.px + b.py + x / (e.px + f.py + g)]
	// [c.px + d.py + y / (e.px + f.py + g)]
	// [                1                  ]

	const Matrix<T,D+1,D+1> &td = this->direct();
	Point<T,D+1> aux;

	static_assert(P::dim == D, "Mismatched dimensions");
	if(p.size()+1 != td.rows())
		throw std::runtime_error("Mismatched dimensions");

	// só há ganho significativo se loop interno for com iterador,
	// c/ o externo não há diferença
	for(int i=0; i<D+1; ++i)
	{
		aux[i] = td[i][D];

		auto ittdi = td[i].begin();
		for(auto itp=p.begin(); itp!=p.end();  ++itp, ++ittdi)
			aux[i] += *ittdi * *itp;
	}

#if 0
	for(int i=0; i<D+1; ++i)
	{
		aux[i] = td[i][D];
		for(int j=0; j<D; ++j)
			aux[i] += td[i][j]*p[j];
	}
#endif

	Point<T,D> ret;
	// gcc-4.5.0 não tá fazendo loop unrolling quando usando iteradores...
#if 0
	auto itaux = aux.begin();
	for(auto itret = ret.begin(); itret != ret.end(); ++itret, ++itaux)
		*itret = *itaux / aux[D];
#endif
	for(int i=0; i<D; ++i)
		ret[i] = aux[i] / aux[D];

	return std::move(ret);
}/*}}}*/

template <class T, int D> 
ProjTransform<T,D> &ProjTransform<T,D>::operator*=(const ProjTransform &xf)/*{{{*/
{
	if(xf)
	{
		if(*this)
		{
			this->direct() *= static_cast<const Matrix<T,D+1,D+1> &>(xf);
			this->modified();
		}
		else
			*this = xf;
	}
	return *this;
}/*}}}*/

template <class T, int D> template <int M, int N>
ProjTransform<T,D> &ProjTransform<T,D>::operator*=(const Matrix<T,M,N> &m)/*{{{*/
{
	if(*this)
	{
		this->direct() *= m;
		this->modified();
	}
	else
		*this = m;
	return *this;
}/*}}}*/

template <class T, int D>
ProjTransform<T,D> &ProjTransform<T,D>::operator*=(const Size<T,D> &s)/*{{{*/
{
	scale_by(s);
	return *this;
}/*}}}*/
template <class T, int D>
ProjTransform<T,D> &ProjTransform<T,D>::operator/=(const Size<T,D> &s)/*{{{*/
{
	resize_by(1/s);
	return *this;
}/*}}}*/

template <class T, int D>
ProjTransform<T,D> &ProjTransform<T,D>::operator+=(const Vector<T,D> &v)/*{{{*/
{
	translate_by(v);
	return *this;
}/*}}}*/
template <class T, int D>
ProjTransform<T,D> &ProjTransform<T,D>::operator-=(const Vector<T,D> &v)/*{{{*/
{
	translate_by(-v);
	return *this;
}/*}}}*/

template <class T, int D>
ProjTransform<T,D> &ProjTransform<T,D>::operator*=(const T &s)/*{{{*/
{
	scale_by(s);
	return *this;
}/*}}}*/
template <class T, int D>
ProjTransform<T,D> &ProjTransform<T,D>::operator/=(const T &s)/*{{{*/
{
	scale_by(1/s);
	return *this;
}/*}}}*/

template <class T, int D>
auto ProjTransform<T,D>::translate_by(const Vector<T,D> &v)/*{{{*/
	-> ProjTransform &
{
	// [a  b  x+a.vx+b.vy]
	// [c  d  y+c.vx+d.vy]
	// [e  f  w+e.vx+f.vy]

	Matrix<T,D+1,D+1> &td = this->direct();
	for(int i=0; i<D+1; ++i)
		for(int j=0; j<D; ++j)
			td[i][D] += td[i][j]*v[j];

	this->modified();
	return *this;
}/*}}}*/
template <class T, int D> template <class... ARGS> 
auto ProjTransform<T,D>::translate_by(const ARGS &...v)/*{{{*/
	-> ProjTransform &
{
	return translate_by(Vector<T,D>(v...));
}/*}}}*/
template <class T, int D>
auto ProjTransform<T,D>::scale_by(const Size<T,D> &s)/*{{{*/
	-> ProjTransform &
{
	// [a.sx  b.sy  x]
	// [c.sx  d.sy  y]
	// [e.sx  f.sy  w]
	
	Matrix<T,D+1,D+1> &td = this->direct();
	for(int i=0; i<D+1; ++i)
		for(int j=0; j<D; ++j)
			td[i][j] *= s[j];

	this->modified();
	return *this;
}/*}}}*/
template <class T, int D> template <class... ARGS> 
auto ProjTransform<T,D>::scale_by(const ARGS &...v)/*{{{*/
	-> ProjTransform &
{
	return scale_by(Size<T,D>(v...));
}/*}}}*/
template <class T, int D> template <class U>
auto ProjTransform<T,D>::rotate_by(const U &r)/*{{{*/
	-> ProjTransform &
{
	return *this *= create_rot(r);
}/*}}}*/
template <class T, int D>
auto ProjTransform<T,D>::rotate_by(const UnitVector<T,D> &axis, T angle)/*{{{*/
	-> ProjTransform &
{
	return rotate_by(AxisAngle<T>(axis, angle));
}/*}}}*/

template <class T, int D>
ProjTransform<T,D> *ProjTransform<T,D>::do_get_inverse() const/*{{{*/
{
	return new ProjTransform<T,D>(inv(this->direct()));
}/*}}}*/

template <class T, int D>
ProjTransform<T,D> *ProjTransform<T,D>::do_get_transpose() const/*{{{*/
{
	return new ProjTransform<T,D>(transpose(this->direct()));
}/*}}}*/

template <class T>
Box<T,2> proj(const Box<T,3> &b, ortho_plane plane)/*{{{*/
{
	switch(plane)
	{
	case PLANE_X:
		return Box<T,2>(Point<T,2>(b.y, b.z), Size<T,2>(b.h, b.d));
	case PLANE_Y:
		return Box<T,2>(Point<T,2>(b.x, b.z), Size<T,2>(b.w, b.d));
	case PLANE_Z:
		return Box<T,2>(Point<T,2>(b.x, b.y), Size<T,2>(b.w, b.h));
	default:
		assert(false);
		return null_box<T,2>();
	}
}/*}}}*/

template <class T>
Point<T,2> proj(const Point<T,3> &p, ortho_plane plane)/*{{{*/
{
	switch(plane)
	{
	case PLANE_X:
		return Point<T,2>(p.y, p.z);
	case PLANE_Y:
		return Point<T,2>(p.x, p.z);
	case PLANE_Z:
		return Point<T,2>(p.x, p.y);
	default:
		assert(false);
		return Point<T,2>(0,0);
	}
}/*}}}*/

template <class T, template <class,int> class V>
auto proj(const V<T,3> &p, ortho_plane plane) /*{{{*/
	-> typename requires<is_vector<V<T,3>>, Vector<T,2>>::type
{
	switch(plane)
	{
	case PLANE_X:
		return Vector<T,2>(p.y, p.z);
	case PLANE_Y:
		return Vector<T,2>(p.x, p.z);
	case PLANE_Z:
		return Vector<T,2>(p.x, p.y);
	default:
		assert(false);
		return Vector<T,2>(0,0);
	}
}/*}}}*/

template <class T>
Size<T,2> proj(const Size<T,3> &s, ortho_plane plane)/*{{{*/
{
	switch(plane)
	{
	case PLANE_X:
		return Size<T,2>(s.h, s.d);
	case PLANE_Y:
		return Size<T,2>(s.w, s.d);
	case PLANE_Z:
		return Size<T,2>(s.w, s.h);
	default:
		assert(false);
		return Size<T,2>(0,0);
	}
}/*}}}*/

template <class T>
Point<T,2> proj(const Point<T,3> &p, const ProjTransform<T,3> &xf)/*{{{*/
{
	Point<T,3> xfp = xf * p;
	return Point<T,2>(xfp.x, xfp.y);
}/*}}}*/

template <class T>
Box<T,2> proj(const Box<T,3> &p, const ProjTransform<T,3> &xf)/*{{{*/
{
	Box<T,3> xfb = xf * p;
	return Box<T,2>(Point<T,2>(xfb.x, xfb.y), Size<T,2>(xfb.w, xfb.h));
}/*}}}*/

template <class T>
Point<T,3> unproj(const Point<T,2> &p, real z, ortho_plane plane)/*{{{*/
{
	switch(plane)
	{
	case PLANE_X:
		return Point<T,3>(z, p.x, p.y);
	case PLANE_Y:
		return Point<T,3>(p.x, z, p.y);
	case PLANE_Z:
		return Point<T,3>(p.x, p.y, z);
	default:
		assert(false);
		return Point<T,3>(0,0,0);
	}
}/*}}}*/

template <class T>
Vector<T,3> unproj(const Vector<T,2> &p, real z, ortho_plane plane)/*{{{*/
{
	switch(plane)
	{
	case PLANE_X:
		return Vector<T,3>(z, p.x, p.y);
	case PLANE_Y:
		return Vector<T,3>(p.x, z, p.y);
	case PLANE_Z:
		return Vector<T,3>(p.x, p.y, z);
	default:
		assert(false);
		return Vector<T,3>(0,0,0);
	}
}/*}}}*/

template <class T>
Point<T,3> unproj(const Point<T,2> &p, T z, const ProjTransform<T,3> &xf)/*{{{*/
{
	return inv(xf) * Point<T,3>(p.x, p.y, z);
}/*}}}*/

template <class T>
Point<T,2> proj_sphere(const Point<T,3> &p)/*{{{*/
{
	return Point<T,2>(atan2(p.x, p.z), atan2(p.y, sqrt(p.x*p.x + p.z*p.z)));
}/*}}}*/

template <class T>
Point<T,3> unproj_sphere(const Point<T,2> &p)/*{{{*/
{
	return Point<T,3>(sin(p.x)*cos(p.y), sin(p.y), cos(p.x)*cos(p.y));
}/*}}}*/

template <class T>
ProjTransform<T,3> persp_proj(const Box<T,2> &wnd, T near, T far)/*{{{*/
{
	T dz = far-near;

	return ProjTransform<T,3>
		{{2*near/wnd.w,            0, (wnd.x+wnd.w*2)/wnd.w, 0}, 
		 {		     0, 2*near/wnd.h, (wnd.y+wnd.h*2)/wnd.h, 0},
		 {		     0,            0, -(near+far)/dz, -2*near*far/dz},
		 {		     0,			   0,  -1,			 		 0}};
}/*}}}*/

template <class T>
ProjTransform<T,3> persp_proj(T fovy, T aspect, T near, T far)/*{{{*/
{
	T f = cos(fovy/2)/sin(fovy/2);
	T dz = near-far;

	return ProjTransform<T,3>
		{{f/aspect, 0,			  0,			 0},
		 {		 0, f,			  0,			 0},
		 {		 0, 0,(far+near)/dz, 2*near*far/dz},
		 {		 0, 0,			 -1,		     0}};
}/*}}}*/

template <class T>
AffineTransform<T,3> ortho_proj(const Box<T,2> &wnd, T near, T far)/*{{{*/
{
	T dz = far-near;

	return AffineTransform<T,3>
		{{2/wnd.w,       0,     0, -(wnd.x+wnd.w*2)/wnd.w},
		 {      0, 2/wnd.h,     0, -(wnd.y+wnd.h*2)/wnd.h},
		 {      0,       0, -2/dz,         -(near+far)/dz}};
}/*}}}*/

}}

// $Id: proj_transform.hpp 2240 2009-06-05 22:49:41Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

