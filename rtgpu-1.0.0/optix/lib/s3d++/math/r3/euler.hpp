namespace s3d { namespace math { namespace r3
{

template <rotation_frame F, class T>
auto Euler<F,T>::operator-() const -> Euler/*{{{*/
{
	return { -theta, -phi, -psi };
}/*}}}*/

template <rotation_frame F, class T>
auto Euler<F,T>::operator+=(const Euler &that) -> Euler&/*{{{*/
{
	theta += that.theta;
	phi += that.phi;
	psi += that.psi;
	return *this;
}/*}}}*/

template <rotation_frame F, class T>
auto Euler<F,T>::operator-=(const Euler &that) -> Euler&/*{{{*/
{
	theta -= that.theta;
	phi -= that.phi;
	psi -= that.psi;
	return *this;
}/*}}}*/

template <rotation_frame F, class T>
bool Euler<F,T>::operator==(const Euler &that) const/*{{{*/
{
	auto r1 = normalize(*this),
		 r2 = normalize(that);

	return equal(r1.theta, r2.theta) && equal(r1.phi, r2.phi) 
		&& equal(r1.psi, r2.psi);
}/*}}}*/

template <rotation_frame F, class T>
Euler<rotframe_traits<F>::revert,T> rev(const Euler<F,T> &r)/*{{{*/
{
	return { r.psi, r.phi, r.theta };
}/*}}}*/

template <rotation_frame F, class T>
std::ostream &operator<<(std::ostream &out, const Euler<F,T> &r)/*{{{*/
{
	char x = 'X'+rotframe_traits<F>::i,
		 y = 'X'+rotframe_traits<F>::j,
		 z = 'X'+rotframe_traits<F>::h;

	if(rotframe_traits<F>::rotating_frame)
		swap(x,z);

	return out << '(' << x << '<' << (equal(r.theta,0)?0:deg(r.theta)) << "°"
			   << ';' << y << '<' << (equal(r.phi,0)?0:deg(r.phi)) << "°"
			   << ';' << z << '<' << (equal(r.psi,0)?0:deg(r.psi)) << "°"
			   << ';' << (rotframe_traits<F>::rotating_frame?'r':'s') 
			   << ')';
}/*}}}*/

template <rotation_frame F, class T>
Euler<F,T> &normalize_inplace(Euler<F,T> &r)/*{{{*/
{
	if(rotframe_traits<F>::odd_parity)
		r.phi = -r.phi;

	if(rotframe_traits<F>::repeating)
	{
		r.phi = mod(r.phi,0,2*M_PI);

		if(greater_than(r.phi,M_PI))
		{
			r.theta += M_PI;
			r.phi = 2*M_PI-r.phi;
			r.psi += M_PI;
		}

		if(equal(r.phi,0))
		{
			r.theta += r.psi;
			r.psi = 0;
		}
		else if(equal(r.phi,M_PI))
		{
			r.theta -= r.psi;
			r.psi = 0;
		}

		assert(greater_or_equal_than(r.phi,T(0)) && 
			   less_or_equal_than(r.phi,T(M_PI)));
	}
	else
	{
		r.phi = mod(r.phi,-M_PI,M_PI);

		if(less_than(r.phi,-M_PI/2))
		{
			r.theta += M_PI;
			r.phi = -M_PI-r.phi;
			r.psi += M_PI;
		}
		else if(greater_than(r.phi,M_PI/2))
		{
			r.theta += M_PI;
			r.phi = M_PI-r.phi;
			r.psi += M_PI;
		}

		if(equal(r.phi,-M_PI/2))
		{
			r.theta += r.psi;
			r.psi = 0;
		}
		else if(equal(r.phi,M_PI/2))
		{
			r.theta -= r.psi;
			r.psi = 0;
		}

		assert(greater_or_equal_than(r.phi,T(-M_PI/2)) && 
			   less_or_equal_than(r.phi,T(M_PI/2)));

	}

	if(rotframe_traits<F>::odd_parity)
		r.phi = -r.phi;

	r.theta = mod(r.theta,-M_PI,M_PI);
	r.psi = mod(r.psi,-M_PI,M_PI);

	assert(greater_or_equal_than(r.theta,T(-M_PI)) && 
		   less_than(r.theta,T(M_PI)));

	assert(greater_or_equal_than(r.psi,T(-M_PI)) && 
		   less_than(r.psi,T(M_PI)));
	return r;
}/*}}}*/

template <rotation_frame F, class T>
Euler<F,T> normalize(Euler<F,T> r)/*{{{*/
{
	return normalize_inplace(r);
}/*}}}*/

inline int main_axis(rotation_frame frame)
{
	return (frame & AXIS) >> 3;
}
inline std::array<int,3> axis(rotation_frame frame)
{
	std::array<int, 3> a;
	a[0] = main_axis(frame);
	a[1] = a[0] + 1;
	if(a[1] == 3)
		a[1] = 0;

	a[2] = a[1] + 1;
	if(a[2] == 3)
		a[2] = 0;

	return a;
}

inline bool odd_parity(rotation_frame frame)
{
	return (frame & PARITY) == PAR_ODD;
}
inline bool repeating(rotation_frame frame)
{
	return (frame & REPETITION) == REP_YES;
}

inline bool rotating(rotation_frame frame)
{
	return (frame & FRAME_DYNAMICS) == FRAME_ROTATING;
}

}}} // namespace s3d::math::r3
