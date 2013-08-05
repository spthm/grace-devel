#ifndef S3D_MATH_R4_QUATERNION_SPACE_H
#define S3D_MATH_R4_QUATERNION_SPACE_H

namespace s3d { namespace math { namespace r4
{

template <class T> struct quaternion_space
{
public:
	static const int dim = 4;
	typedef T value_type;

	template <class U>
	quaternion_space &operator=(const quaternion_space<U> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	quaternion_space &operator=(const quaternion_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &w; }
	iterator end() { return &w + dim; }

	const_iterator begin() const { return &w; }
	const_iterator end() const { return &w + dim; }

	T w, x, y, z;
};

}}} // namespace s3d::math::r4


#endif
