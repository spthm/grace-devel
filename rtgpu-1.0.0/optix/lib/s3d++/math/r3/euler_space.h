#ifndef S3D_MATH_R3_EULER_SPACE_H
#define S3D_MATH_R3_EULER_SPACE_H

namespace s3d { namespace math { namespace r3
{

template <class T> class euler_space
{
public:
	static const int dim = 3;
	typedef T value_type;

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &theta; }
	iterator end() { return &theta + dim; }

	const_iterator begin() const { return &theta; }
	const_iterator end() const { return &theta + dim; }

	T theta, phi, psi;
};


}}} // namesapce s3d::math::r3

#endif
