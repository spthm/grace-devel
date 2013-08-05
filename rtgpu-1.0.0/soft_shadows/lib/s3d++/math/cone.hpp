namespace s3d { namespace math
{

template <class T, int D>
bool Cone<T,D>::is_coupled_to(const Vector<T,D> &v) const
{
	if(dot(dir, -v) > cosa)
		return true;
	else
		return false;
}

template <class T, int D>
bool are_coupled(const Cone<T,D> &c1, const Cone<T,D> &c2)
{
	auto d = unit(c1.origin - c2.origin);
	return c1.is_coupled_to(d) && c2.is_coupled_to(-d);
}


}} // namespace s3d::math
