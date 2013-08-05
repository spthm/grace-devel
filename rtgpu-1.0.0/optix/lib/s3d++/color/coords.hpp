namespace s3d { namespace color
{

template <class C, class S>
auto coords<C,S>::operator+=(const color_type &that) -> color_type &/*{{{*/
{
	assert(size() == that.size());

	auto its = that.begin(); auto itd = begin();
	while(itd != end())
		*itd++ += *its++;

	return static_cast<color_type &>(*this);
}/*}}}*/
template <class C, class S>
auto coords<C,S>::operator-=(const color_type &that) -> color_type &/*{{{*/
{
	auto its = that.begin(); auto itd = begin();
	while(itd != end())
		*itd++ -= *its++;

	return static_cast<color_type &>(*this);
}/*}}}*/
template <class C, class S>
auto coords<C,S>::operator*=(const color_type &that) -> color_type &/*{{{*/
{
	auto its = that.begin(); auto itd = begin();
	while(itd != end())
		*itd++ *= *its++;

	return static_cast<color_type &>(*this);
}/*}}}*/
template <class C, class S>
auto coords<C,S>::operator/=(const color_type &that) -> color_type &/*{{{*/
{
	auto its = that.begin(); auto itd = begin();
	while(itd != end())
		*itd++ /= *its++;

	return static_cast<color_type &>(*this);
}/*}}}*/

template <class C, class S> template <class U>
auto coords<C,S>::operator+=(const U &v)/*{{{*/
	-> typename std::enable_if<std::is_convertible<U,value_type>::value &&
							   // arithmetic types can be converted to RADIANCE
							   // implicitly
	                           traits::model<C>::value!=model::RADIANCE,
							   color_type&>::type
{
	for(auto it=begin(); it!=end(); ++it)
		*it += v;
	return static_cast<color_type &>(*this);
}/*}}}*/

template <class C, class S> template <class U>
auto coords<C,S>::operator-=(const U &v)/*{{{*/
	-> typename std::enable_if<std::is_convertible<U,value_type>::value &&
							   // arithmetic types can be converted to RADIANCE
							   // implicitly
	                           traits::model<C>::value!=model::RADIANCE,
							   color_type&>::type
{
	for(auto it=begin(); it!=end(); ++it)
		*it -= v;
	return static_cast<color_type &>(*this);
}/*}}}*/

template <class C, class S> template <class U>
auto coords<C,S>::operator*=(const U &v)/*{{{*/
	-> typename std::enable_if<std::is_convertible<U,value_type>::value &&
							   // arithmetic types can be converted to RADIANCE
							   // implicitly
	                           traits::model<C>::value!=model::RADIANCE,
							   color_type&>::type
{
	for(auto it=begin(); it!=end(); ++it)
		*it *= v;
	return static_cast<color_type &>(*this);
}/*}}}*/
template <class C, class S> template <class U>
auto coords<C,S>::operator/=(const U &v)/*{{{*/
	-> typename std::enable_if<std::is_convertible<U,value_type>::value &&
							   // arithmetic types can be converted to RADIANCE
							   // implicitly
	                           traits::model<C>::value!=model::RADIANCE,
							   color_type&>::type
{
	for(auto it=begin(); it!=end(); ++it)
		*it /= v;
	return static_cast<color_type &>(*this);
}/*}}}*/

}} // namespace s3d::color
