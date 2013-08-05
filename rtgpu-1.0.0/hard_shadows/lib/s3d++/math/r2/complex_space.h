#ifndef S3D_MATH_R2_COMPLEX_SPACE_H
#define S3D_MATH_R2_COMPLEX_SPACE_H

namespace s3d { namespace math { namespace r2
{

template <class T> class complex_space
{
public:
	static const int dim = 2;
	typedef T value_type;

	template <class U>
	complex_space &operator=(const complex_space<U> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	complex_space &operator=(const complex_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &re; }
	iterator end() { return &re+dim; }

	const_iterator begin() const { return &re; }
	const_iterator end() const { return &re+dim; }

	T re, im;
};


}}} // namespace s3d::math::r2

#endif
