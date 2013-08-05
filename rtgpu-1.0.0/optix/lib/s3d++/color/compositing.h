#ifndef S3D_COLOR_COMPOSITING_H
#define S3D_COLOR_COMPOSITING_H

namespace s3d { namespace color
{

template <class C>
auto alpha_channel(const C &c) -> real
{
	return 1;
}


template <class C>
class compositing_operators
{
public:
	friend C over(const C &c1, const C &c2)
	{
		return comp(c1, 1, c2, 1-alpha_channel(c1));
	}
	C over(const C &c2) const
		{ return over(*this, c2); }

	friend C in(const C &c1, const C &c2)
	{
		return comp(c1, alpha_channel(c2), c2, 0);
	}
	C in(const C &c2) const
		{ return in(*this, c2); }

	friend C out(const C &c1, const C &c2)
	{
		return comp(c1, 1-alpha_channel(c2), c2, 0);
	}
	C out(const C &c2) const
		{ return out(*this, c2); }

	friend C atop(const C &c1, const C &c2)
	{
		return comp(c1, 1-alpha_channel(c2), c2, alpha_channel(c1));
	}
	C atop(const C &c2) const
		{ return atop(*this, c2); }

	friend C operator ^(const C &c1, const C &c2)
	{
		return comp(c1, 1-alpha_channel(c2), c2, 1-alpha_channel(c1));
	}
	C &operator ^=(const C &c2)
	{
		*this = *this ^ c2;
		return *this;
	}
};

template <class C>
C comp(const C &c1, real f1, const C &c2, real f2)
{
	assert(greater_or_equal_than(f1,0) && less_or_equal_than(f1,1));
	assert(greater_or_equal_than(f2,0) && less_or_equal_than(f2,1));

	return C(c1*f1 + c2*f2);
}

}} // namespace s3d::color

#endif
