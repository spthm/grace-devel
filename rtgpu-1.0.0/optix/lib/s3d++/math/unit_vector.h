#ifndef S3D_MATH_UNIT_VECTOR_H
#define S3D_MATH_UNIT_VECTOR_H

#include "vector.h"

namespace s3d { namespace math
{

template <class T, int D> struct UnitVector
	: coords<UnitVector<T,D>,euclidean_space<const T,D>>
{
	static_assert(std::is_floating_point<
		typename value_type<T,order<T>::value>::type>::value, 
		"Must be a floating point type");

private:
	typedef coords<UnitVector,euclidean_space<const T,D>> coords_base;
public:
	UnitVector(); // so that it can be in a std::tuple

	template <class U>
	UnitVector(const UnitVector<U,D> &c) : coords_base(c) {}

	UnitVector(const UnitVector &that) = default;

	template <class... ARGS, class = 
		typename std::enable_if<D==RUNTIME || D==sizeof...(ARGS)+1>::type>
	UnitVector(T v1, ARGS ...vn);

	UnitVector(const Vector<T,D> &c, bool is_unit);
	explicit UnitVector(const Vector<T,D> &c);

	UnitVector &operator=(const UnitVector &that);

	template <class U>
	operator Vector<U,D> () const;

	UnitVector operator-() const;

	UnitVector &operator*=(const UnitVector &that);
	UnitVector &operator/=(const UnitVector &that);

	using coords_base::begin;
	using coords_base::end;
};

template <class T, int D>
struct is_vector<UnitVector<T,D>>
{
	static const bool value = true;
};

}} // namespace s3d::math

#include "unit_vector.hpp"

#endif
