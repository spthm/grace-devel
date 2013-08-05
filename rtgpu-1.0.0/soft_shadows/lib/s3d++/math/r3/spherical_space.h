#ifndef S3D_MATH_R3_SPHERICAL_SPACE_H
#define S3D_MATH_R3_SPHERICAL_SPACE_H

namespace s3d { namespace math { namespace r3
{

template <class T, class A> struct spherical_space
{
	static const int dim = 3;
	typedef T value_type;

	T r;
	A theta, phi;
};

} // namespace r3

template <size_t I, class T> struct element;

template <class T, class A>
struct element<0, r3::spherical_space<T,A>>
{
	typedef T type;

	static type &get(r3::spherical_space<T,A> &p) 
		{ return p.r; }
	static const type &get(const r3::spherical_space<T,A> &p) 
		{ return p.r; }
};

template <class T, class A>
struct element<1, r3::spherical_space<T,A>>
{
	typedef A type;

	static type &get(r3::spherical_space<T,A> &p) 
		{ return p.theta; }
	static const type &get(const r3::spherical_space<T,A> &p) 
		{ return p.theta; }
};

template <class T, class A>
struct element<2, r3::spherical_space<T,A>>
{
	typedef A type;

	static type &get(r3::spherical_space<T,A> &p) 
		{ return p.phi; }
	static const type &get(const r3::spherical_space<T,A> &p) 
		{ return p.phi; }
};

}} // namespace s3d::math

#endif
