namespace s3d { namespace math { namespace r2
{

template <class T=real, class A>
UnitComplex<T> to_complex(const A &ang)/*{{{*/
{
	return UnitComplex<T>{ { T(cos(ang)), T(sin(ang)) }, true };
}/*}}}*/

template <class T> 
Matrix<T,2,2> to_rot_matrix(const UnitComplex<T> &c)/*{{{*/
{
	return Matrix<T,2,2>{ { c.re, -c.im}, 
						  { c.im,  c.re} };
}/*}}}*/

template <class T=real, class A> 
Matrix<T,2,2> to_rot_matrix(const A &ang)/*{{{*/
{
	return to_rot_matrix(to_complex<T>(ang));
}/*}}}*/

template <class A=real, class T>
A angle(const UnitComplex<T> &c)/*{{{*/
{
	return atan2(c.im, c.re);
}/*}}}*/

template <class T>
bool is_identity(const UnitComplex<T> &c)/*{{{*/
{
	return equal(c.re, 1) && equal(c.im, 0);
}/*}}}*/

template <class T>
const UnitComplex<T> &normalize(const UnitComplex<T> &c)/*{{{*/
{
	return c;
}/*}}}*/

}}} // s3d::math::r2
