namespace s3d { namespace math { namespace r3
{

template <class T, class A>
bool AxisAngle<T,A>::operator==(const AxisAngle &that) const/*{{{*/
{
	auto aa1 = normalize(*this),
		 aa2 = normalize(that);

	return aa1.axis == aa2.axis && equal(aa1.angle, aa2.angle);
}/*}}}*/
template <class T, class A>
auto AxisAngle<T,A>::operator-() const -> AxisAngle/*{{{*/
{
	return { -axis, -angle };
}/*}}}*/
template <class T, class A>
auto AxisAngle<T,A>::operator+=(const AxisAngle &that) -> AxisAngle &/*{{{*/
{
	axis += that.axis;
	angle += that.angle;
	return *this;
}/*}}}*/
template <class T, class A>
auto AxisAngle<T,A>::operator-=(const AxisAngle &that) -> AxisAngle &/*{{{*/
{
	axis -= that.axis;
	angle -= that.angle;
	return *this;
}/*}}}*/

template <class T, class A>
A angle(const AxisAngle<T,A> &aa)/*{{{*/
{
	return aa.angle;
}/*}}}*/

template <class T, class A>
UnitVector<T,3> axis(const AxisAngle<T,A> &aa)/*{{{*/
{
	return aa.axis;
}/*}}}*/

template <class T, class A>
AxisAngle<T,A> &normalize_inplace(AxisAngle<T,A> &aa)/*{{{*/
{
	aa.angle = mod(aa.angle, -M_PI, M_PI);

	if(equal(aa.angle, 0))
		aa.axis = {0,0,1};
	else
	{
		if(equal(aa.angle,-M_PI))
		{
			auto d = dot(aa.axis, r3::vector{0,0,1});
			if(equal(d,0))
			{
				d = dot(aa.axis, r3::vector{0,1,0});
				if(equal(d,0))
					d = dot(aa.axis, r3::vector{1,0,0});
			}

			if(d < 0)
			{
				aa.axis = -aa.axis;
				aa.angle = mod(-aa.angle, -M_PI, M_PI);
			}
		}
		else if((aa.angle) < 0)
		{
			aa.angle = mod(-aa.angle, -M_PI, M_PI);
			aa.axis = -aa.axis;
		}
	}

	return aa;
}/*}}}*/

template <class T, class A>
AxisAngle<T,A> normalize(AxisAngle<T,A> aa)/*{{{*/
{
	return normalize_inplace(aa);
}/*}}}*/

template <class T, class A>
AxisAngle<T,A> inv(const AxisAngle<T,A> &aa)/*{{{*/
{
	return {-aa.axis, aa.angle};
}/*}}}*/

template <class T, class A>
AxisAngle<T,A> &inv_inplace(AxisAngle<T,A> &aa)/*{{{*/
{
	aa.axis = -aa.axis;
	return aa;
}/*}}}*/

template <class T, class A>
std::ostream &operator<<(std::ostream &out, const AxisAngle<T,A> &aa)/*{{{*/
{
	return out << aa.axis << ';' << deg(aa.angle) << "°";
}/*}}}*/

}}} // namespace s3d::math::r3
