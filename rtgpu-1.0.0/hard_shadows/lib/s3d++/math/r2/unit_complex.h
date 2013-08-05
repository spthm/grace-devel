#ifndef S3D_MATH_R2_UNIT_COMPLEX_H
#define S3D_MATH_R2_UNIT_COMPLEX_H

#include "complex.h"

namespace s3d { namespace math { namespace r2
{

template <class T> struct UnitComplex
	: coords<UnitComplex<T>,complex_space<const T>>
{
	static_assert(std::is_floating_point<
		typename value_type<T,order<T>::value>::type>::value, 
		"Must be a floating point type");
private:
	typedef coords<UnitComplex,complex_space<const T>> coords_base;
public:
	using coords_base::re;
	using coords_base::im;

	template <class U>
	UnitComplex(const UnitComplex<U> &c) : coords_base(c) {}

	explicit UnitComplex(const Complex<T> &c);
	UnitComplex(const Complex<T> &c, bool is_unit);

	operator const Complex<T> &() const;
};

template <class T>
struct is_complex<UnitComplex<T>>
{
	static const bool value = true;
};

template <class T, int D, class U=typename make_floating_point<T>::type>
auto unit(Complex<T> c, U *norm=NULL) -> UnitComplex<T>;

template <class T> 
auto unit(const UnitComplex<T> &c) -> const UnitComplex<T> &;

template <class T> 
auto is_unit(const UnitComplex<T> &c) -> bool;

template <class T> T abs(const UnitComplex<T> &c);
template <class T> T norm(const UnitComplex<T> &c);
template <class T> T sqrnorm(const UnitComplex<T> &c);

}}} // namespace s3d::math::r2

#include "unit_complex.hpp"

#endif
