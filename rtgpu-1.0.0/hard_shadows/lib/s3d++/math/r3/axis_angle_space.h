#ifndef S3D_MATH_R3_AXIS_ANGLE_SPACE_H
#define S3D_MATH_R3_AXIS_ANGLE_SPACE_H

namespace s3d { namespace math { namespace r3
{

template <class T, class A>
struct axis_angle_space
{
	static const int dim = 2;

	UnitVector<T,3> axis;
	A angle;
};

} // namespace r3

template <size_t I, class T> struct element;

template <class T, class A>
struct element<0, r3::axis_angle_space<T,A>>
{
	typedef UnitVector<T,3> type;

	static type &get(r3::axis_angle_space<T,A> &p) 
		{ return p.axis; }
	static const type &get(const r3::axis_angle_space<T,A> &p) 
		{ return p.axis; }
};

template <class T, class A>
struct element<1, r3::axis_angle_space<T,A>>
{
	typedef A type;

	static type &get(r3::axis_angle_space<T,A> &p) 
		{ return p.angle; }
	static const type &get(const r3::axis_angle_space<T,A> &p) 
		{ return p.angle; }
};

}} // namespace s3d::math

#endif
