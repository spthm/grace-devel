namespace s3d { namespace math
{

namespace r4
{

template <class T>
UnitQuaternion<T>::UnitQuaternion(const Quaternion<T> &q, bool _is_unit) /*{{{*/
	: coords_base(_is_unit ? reinterpret_cast<const UnitQuaternion &>(q) : unit(q))
{
	assert(is_unit(static_cast<const Quaternion<T>&>(*this)));
}/*}}}*/

template <class T> template <class... ARGS> 
UnitQuaternion<T>::UnitQuaternion(T c1, ARGS... cn)/*{{{*/
	: coords_base(unit(Quaternion<T>(c1, cn...)))
{
}/*}}}*/

template <class T>
UnitQuaternion<T>::operator const Quaternion<T> &() const/*{{{*/
{
	return reinterpret_cast<const Quaternion<T> &>(*this);
}/*}}}*/

template <class T> 
auto unit(Quaternion<T> q) -> UnitQuaternion<T>/*{{{*/
{ 
	auto n2 = sqrnorm(q);
	if(equal(n2, 1))
		return UnitQuaternion<T>{ q, true };
	else if(n2 == 0)
		return UnitQuaternion<T>{ {1,0,0,0}, true };
	else if(isinf(n2))
		return UnitQuaternion<T>{ {1,0,0,0}, true };
	else
		return UnitQuaternion<T>{ q/=math::sqrt(n2), true };
}/*}}}*/

template <class T>
auto UnitQuaternion<T>::operator *=(const UnitQuaternion &that) /*{{{*/
	-> UnitQuaternion &
{
	// Jeito certo de fazer reinterpret_casts sem quebrar as regras aliasing 
	union conv
	{
		UnitQuaternion<T> *uq;
		Quaternion<T> *q;
	};

	auto &q = *conv{this}.q;
	q *= that;

	auto n = norm(q);
	if(!equal(n,1))
		q /= n;

	assert(is_unit(q));
	return *this;
}/*}}}*/

template <class T>
auto UnitQuaternion<T>::operator /=(const UnitQuaternion &that) /*{{{*/
	-> UnitQuaternion &
{
	auto &q = reinterpret_cast<Quaternion<T>&>(*this);
	q /= that;

	auto n = norm(q);
	if(!equal(n,1))
		q /= n;

	assert(is_unit(q));
	return *this;
}/*}}}*/

template <class T> 
auto unit(const UnitQuaternion<T> &q) -> const UnitQuaternion<T> &/*{{{*/
{
	return q;
}/*}}}*/

template <class T> 
bool is_unit(const UnitQuaternion<T> &q)/*{{{*/
{
	return true;
}/*}}}*/

template <class T, class U> 
auto slerp(const UnitQuaternion<T> &q1, const UnitQuaternion<T> &q2, U t)/*{{{*/
	-> UnitQuaternion<T> 
{
	T cos_theta = dot(q2,q1),
	  theta;

	if(cos_theta + 1 <= std::numeric_limits<T>::epsilon())
		theta = M_PI;
	else if(cos_theta - 1 >= std::numeric_limits<T>::epsilon())
		theta = 0;
	else
		theta = acos(cos_theta);

	T sin_theta = sin(theta);
	if(abs(sin_theta) <= std::numeric_limits<T>::epsilon())
		return theta<M_PI/2 ? q1 : q2;

	T u = sin((1-t)*theta)/sin_theta,
	  v = sin(t*theta)/sin_theta;

	return unit(q1*u + q2*v);
}/*}}}*/

template <class T> 
auto normalize_inplace(UnitQuaternion<T> &q) -> UnitQuaternion<T> &/*{{{*/
{
	if(q.w < 0)
		q = -q;

	return q;
}/*}}}*/

template <class T> 
auto normalize(UnitQuaternion<T> q) -> UnitQuaternion<T>/*{{{*/
{
	return normalize_inplace(q);
}/*}}}*/

// Trivial functions
template <class T> T abs(const UnitQuaternion<T> &q)/*{{{*/
{
	return 1;
}/*}}}*/
template <class T> T norm(const UnitQuaternion<T> &q)/*{{{*/
{
	return 1;
}/*}}}*/
template <class T> T sqrnorm(const UnitQuaternion<T> &q)/*{{{*/
{
	return 1;
}/*}}}*/

}}} // namespace s3d::math::r4
