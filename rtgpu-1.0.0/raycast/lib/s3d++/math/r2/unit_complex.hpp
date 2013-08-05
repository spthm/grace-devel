namespace s3d { namespace math { 

// Result type of arithmetic operations {{{
template <class T, class U>
struct result_add<UnitComplex<T>, UnitComplex<U>>
	: result_add_dispatch<Complex<T>, Complex<U>>
{
};

template <class T, class U>
struct result_sub<UnitComplex<T>, UnitComplex<U>> 
	: result_sub_dispatch<Complex<T>, Complex<U>>
{
};

template <class T, class U>
struct result_mul<UnitComplex<T>, UnitComplex<U>>
	: result_mul_dispatch<Complex<T>, Complex<U>>
{
};

template <class T, class U>
struct result_div<UnitComplex<T>, UnitComplex<U>>
	: result_div_dispatch<Complex<T>, Complex<U>>
{
};

template <class T, int M, class U>
struct result_mul<Matrix<T,M>,UnitComplex<U>> 
	: result_mul_dispatch<Matrix<T,M>, Complex<U>>
{
};

template <class T, class U>
struct result_add<UnitComplex<T>, arithmetic<U>> 
	: result_add_dispatch<Complex<T>,U>
{
};

template <class T, class U>
struct result_sub<UnitComplex<T>, arithmetic<U>> 
	: result_sub_dispatch<Complex<T>,U>
{
};

template <class T, class U>
struct result_mul<UnitComplex<T>, arithmetic<U>> 
	: result_mul_dispatch<Complex<T>,U>
{
};

template <class T, class U>
struct result_div<UnitComplex<T>, arithmetic<U>> 
	: result_div_dispatch<Complex<T>,U>
{
};

template <class T, class U>
struct result_add<arithmetic<T>, UnitComplex<U>> 
	: result_add_dispatch<T,Complex<U>>
{
};

template <class T, class U>
struct result_sub<arithmetic<T>, UnitComplex<U>> 
	: result_sub_dispatch<T,Complex<U>>
{
};

template <class T, class U>
struct result_mul<arithmetic<T>, UnitComplex<U>> 
	: result_mul_dispatch<T,Complex<U>>
{
};

template <class T, class U>
struct result_div<arithmetic<T>, UnitComplex<U>> 
	: result_div_dispatch<T,Complex<U>>
{
};/*}}}*/
	
namespace r2
{

template <class T> 
UnitComplex<T>::UnitComplex(const Complex<T> &c)/*{{{*/
	: coords_base(unit(c))
{
}/*}}}*/

template <class T> 
UnitComplex<T>::UnitComplex(const Complex<T> &c, bool _is_unit)/*{{{*/
	: coords_base(_is_unit ? reinterpret_cast<const UnitComplex<T> &>(c) : unit(c))
{
	assert(is_unit(static_cast<const Complex<T> &>(*this)));
}/*}}}*/

template <class T> 
UnitComplex<T>::operator const Complex<T>&() const/*{{{*/
{
	return reinterpret_cast<const Complex<T> &>(*this);
}/*}}}*/

template <class T, class U=typename make_floating_point<T>::type>
auto unit(Complex<T> c, U *_norm=NULL) -> UnitComplex<T>/*{{{*/
{
	auto n2 = sqrnorm(c);
	assert(n2 != 0);

	if(equal(n2,1))
	{
		if(_norm)
			*_norm = 1;

		return { c, true };
	}
	else
	{
		auto n = sqrt(n2);

		if(_norm)
			*_norm = n;

		return { c / n, true };
	}
}/*}}}*/

template <class T> 
auto unit(const UnitComplex<T> &c) -> const UnitComplex<T> &/*{{{*/
{
	return c;
}/*}}}*/

template <class T> 
auto is_unit(const UnitComplex<T> &c) -> bool/*{{{*/
{
	return true;
}/*}}}*/

template <class T> 
T abs(const UnitComplex<T> &c)/*{{{*/
{
	return 1;
}/*}}}*/
template <class T> 
T norm(const UnitComplex<T> &c)/*{{{*/
{
	return 1;
}/*}}}*/
template <class T> 
T sqrnorm(const UnitComplex<T> &c)/*{{{*/
{
	return 1;
}/*}}}*/

}}}
