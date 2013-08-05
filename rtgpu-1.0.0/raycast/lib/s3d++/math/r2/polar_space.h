#ifndef S3D_MATH_R2_POLAR_SPACE_H
#define S3D_MATH_R2_POLAR_SPACE_H

namespace s3d { namespace math { namespace r2
{

template <class T, class A> struct polar_space
{
	static const int dim = 2;
	typedef T value_type;

	T r;
	A theta;
};

} // namespace r2

template <size_t I, class T> struct element;

template <class T, class A>
struct element<0, r2::polar_space<T,A>>
{
	typedef T type;

	static type &get(r2::polar_space<T,A> &p) 
		{ return p.r; }
	static const type &get(const r2::polar_space<T,A> &p) 
		{ return p.r; }
};

template <class T, class A>
struct element<1, r2::polar_space<T,A>>
{
	typedef A type;

	static type &get(r2::polar_space<T,A> &p) 
		{ return p.theta; }
	static const type &get(const r2::polar_space<T,A> &p) 
		{ return p.theta; }
};

}} // namespace s3d::math

#endif
