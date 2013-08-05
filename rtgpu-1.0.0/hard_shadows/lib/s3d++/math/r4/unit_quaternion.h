#ifndef S3D_MATH_R4_UNIT_QUATERNION_H
#define S3D_MATH_R4_UNIT_QUATERNION_H

#include "quaternion.h"

namespace s3d { namespace math { namespace r4
{

template <class T>
struct UnitQuaternion
	: coords<UnitQuaternion<T>,quaternion_space<const T>>
{
	static_assert(std::is_floating_point<
		typename value_type<T,order<T>::value>::type>::value, 
		"Must be a floating point type");
private:
	typedef coords<UnitQuaternion,quaternion_space<const T>> coords_base;
public:
	using coords_base::w;
	using coords_base::x;
	using coords_base::y;
	using coords_base::z;

	UnitQuaternion() : coords_base(1,0,0,0) {} // needed sometimes

	template <class U> 
	UnitQuaternion(const UnitQuaternion<U> &that) : coords_base(that) {}

	template <class... ARGS> 
	UnitQuaternion(T c1, ARGS... cn);

	explicit UnitQuaternion(const Quaternion<T> &q, bool is_unit=false);

	operator const Quaternion<T> &() const;

	UnitQuaternion &operator *=(const UnitQuaternion &that);
	UnitQuaternion &operator /=(const UnitQuaternion &that);
};

template <class T>
struct is_quaternion<UnitQuaternion<T>>
{
	static const bool value = true;
};

template <class T> 
auto unit(Quaternion<T> q) -> UnitQuaternion<T>;

template <class T> 
auto unit(const UnitQuaternion<T> &q) -> const UnitQuaternion<T> &;

template <class T> 
bool is_unit(const UnitQuaternion<T> &q);

template <class T, class U> 
auto slerp(const UnitQuaternion<T> &q1, const UnitQuaternion<T> &q2, U t)
	-> UnitQuaternion<T>;

template <class T> 
auto normalize_inplace(UnitQuaternion<T> &q) -> UnitQuaternion<T> &;

template <class T> 
auto normalize(UnitQuaternion<T> q) -> UnitQuaternion<T>;

// Trivial functions
template <class T> T abs(const UnitQuaternion<T> &q);
template <class T> T norm(const UnitQuaternion<T> &q);
template <class T> T sqrnorm(const UnitQuaternion<T> &q);

}}} // namespace s3d::math::r4

#include "unit_quaternion.hpp"

#endif
