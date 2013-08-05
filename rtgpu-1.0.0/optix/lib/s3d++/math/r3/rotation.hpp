#include "../matrix.h"
#include "../r4/unit_quaternion.h"
#include "axis_angle.h"
#include "euler.h"
#include "../unit_vector.h"
#include "../point.h"

namespace s3d { namespace math { namespace r3
{

// From euler.h
template <rotation_frame F, class T>
template <rotation_frame F2, class U>
Euler<F,T>::Euler(const Euler<F2,U> &that)/*{{{*/
	: coords_base(to_euler<F>(to_rot_matrix(that)))
{
}/*}}}*/

// Conversion to quaternion representation

template <rotation_frame F, class T>
UnitQuaternion<T> to_quaternion(const Euler<F,T> &rot)/*{{{*/
{
	auto ti = rot.theta*0.5,
		 tj = rot.phi*0.5,
		 th = rot.psi*0.5;

	if(rotframe_traits<F>::rotating)
		swap(ti, th);

	if(rotframe_traits<F>::odd_parity)
		tj = -tj;

	auto ci = cos(ti), si = sin(ti),
		 cj = cos(tj), sj = sin(tj),
		 ch = cos(th), sh = sin(th);

	auto cc = ci*ch,
		 cs = ci*sh,
		 sc = si*ch,
		 ss = si*sh;

	Quaternion<T> q;
	const int i = rotframe_traits<F>::i+1,
		      j = rotframe_traits<F>::j+1,
		      k = rotframe_traits<F>::k+1;

	if(rotframe_traits<F>::repeating)
	{
		// XYXs
		q[i] = cj*(cs + sc);
		q[j] = sj*(cc + ss);
		q[k] = sj*(cs - sc);
		q[0] = cj*(cc - ss);
	}
	else
	{
		// XYZs
		q[i] = cj*sc - sj*cs;
		q[j] = cj*ss + sj*cc;
		q[k] = cj*cs - sj*sc;
		q[0] = cj*cc + sj*ss;
	}

	if(rotframe_traits<F>::odd_parity)
		q[j] = -q[j];

	return normalize(unit(q));
}/*}}}*/

template <class T, class A> 
UnitQuaternion<T> to_quaternion(const AxisAngle<T,A> &aa)/*{{{*/
{
	T s = sin(aa.angle/2),
	  c = cos(aa.angle/2);

	Quaternion<T> q{c, aa.axis.x*s, aa.axis.y*s, aa.axis.z*s};

	return normalize(unit(q));
}/*}}}*/

template <class T, class A> 
UnitQuaternion<T> to_quaternion(const UnitVector<T,3> &axis, A angle)/*{{{*/
{
	return to_quaternion(AxisAngle<T,A>(axis,angle));
}/*}}}*/

template <class T>
UnitQuaternion<T> to_quaternion(const Matrix<T,3,3> &m)/*{{{*/
{
#if 0
	Quaternion<T> q{ sqrt(max<T>(0, 1 + m[0][0] + m[1][1] + m[2][2]))/2,
					 sqrt(max<T>(0, 1 + m[0][0] - m[1][1] - m[2][2]))/2,
					 sqrt(max<T>(0, 1 - m[0][0] + m[1][1] - m[2][2]))/2,
					 sqrt(max<T>(0, 1 - m[0][0] - m[1][1] + m[2][2]))/2 };

	q.x = copysign(q.x, m[2][1] - m[1][2]);
	q.y = copysign(q.y, m[0][2] - m[2][0]);
	q.z = copysign(q.z, m[1][0] - m[0][1]);
#endif


#if 1
	T tr = trace(m);

	Quaternion<T> q;

	if(greater_than(tr,0))
	{
		T s = sqrt(tr+1),
		  is = 0.5/s;

		q = { s/2, (m[2][1] - m[1][2])*is, 
			       (m[0][2] - m[2][0])*is, 
				   (m[1][0] - m[0][1])*is };
	}
	else
	{
		int i=0;
		if(m[1][1] > m[0][0])
			i = 1;
		if(m[2][2] > m[i][i])
			i = 2;

		static const int next[3] = {1, 2, 0};

		int j = next[i],
			k = next[j];

		T s = sqrt((m[i][i] - (m[j][j] + m[k][k])) + 1);

		if(equal(s,0))
			return {1,0,0,0};

		T is = 0.5/s;


		q.w = (m[k][j] - m[j][k]) * is;
		q[i+1] = s*0.5;
		q[j+1] = (m[i][j] + m[j][i]) * is;
		q[k+1] = (m[i][k] + m[k][i]) * is;
	}
#endif
	return normalize(unit(q));
}/*}}}*/

template <class T> 
UnitQuaternion<T> to_quaternion(const UnitVector<T,3> &from_axis, /*{{{*/
								const UnitVector<T,3> &_from_up, 
								const UnitVector<T,3> &to_axis, 
								const UnitVector<T,3> &_to_up)
{
	auto to_left = unit(cross(_to_up, to_axis)),
		 to_up   = cross(to_axis, to_left),

		 from_left = unit(cross(_from_up, from_axis)),
		 from_up   = cross(from_axis, from_left);

	auto cross_sum = cross(from_left,to_left) +
					 cross(from_up,to_up) +
					 cross(from_axis,to_axis);
	T dot_sum = dot(from_left,to_left) +
		        dot(from_up,to_up) +
				dot(from_axis,to_axis);

	Quaternion<T> q{dot_sum+1,cross_sum.x,cross_sum.y,cross_sum.z};

	return normalize(unit(q));
}/*}}}*/

// Conversion to matrix representation

template <rotation_frame F, class T>
Matrix<T,3,3> to_rot_matrix(const Euler<F,T> &rot)/*{{{*/
{
	auto ti = rot.theta,
		 tj = rot.phi,
		 th = rot.psi;

	if(rotframe_traits<F>::rotating)
		swap(ti, th);

	if(rotframe_traits<F>::odd_parity)
	{
		ti = -ti;
		tj = -tj;
		th = -th;
	}

	auto ci = cos(ti), si = sin(ti),
		 cj = cos(tj), sj = sin(tj),
		 ch = cos(th), sh = sin(th);

	auto cc = ci*ch, 
		 cs = ci*sh,
		 sc = si*ch,
		 ss = si*sh;

	Matrix<T,3,3> m;

	const int i = rotframe_traits<F>::i,
		      j = rotframe_traits<F>::j,
		      k = rotframe_traits<F>::k;

	if(rotframe_traits<F>::repeating)
	{
		// XYXs
		m[i][i] = cj;
		m[i][j] = sj*si;
		m[i][k] = sj*ci;

		m[j][i] = sj*sh;
		m[j][j] = -cj*ss+cc;
		m[j][k] = -cj*cs-sc;

		m[k][i] = -sj*ch;
		m[k][j] = cj*sc+cs;
		m[k][k] = cj*cc-ss;
	}
	else
	{
		// XYZs
		m[i][i] = cj*ch;
		m[i][j] = sj*sc-cs;
		m[i][k] = sj*cc+ss;

		m[j][i] = cj*sh;
		m[j][j] = sj*ss+cc;
		m[j][k] = sj*cs-sc;

		m[k][i] = -sj;
		m[k][j] = cj*si;
		m[k][k] = cj*ci;
	}

	return m;
}/*}}}*/

template <class T, class A>
Matrix<T,3,3> to_rot_matrix(const AxisAngle<T,A> &aa)/*{{{*/
{
	return to_rot_matrix(to_quaternion(aa));
}/*}}}*/

// Conversion to euler representation

template <rotation_frame F, class T>
Euler<F,T> to_euler(const Matrix<T,3,3> &m)/*{{{*/
{
	const int i = rotframe_traits<F>::i,
		      j = rotframe_traits<F>::j,
		      k = rotframe_traits<F>::k;

	Euler<F,T> e;

	if(rotframe_traits<F>::repeating)
	{
		// XYXs
		auto sy = sqrt(m[i][j]*m[i][j] + m[i][k]*m[i][k]);
		if(greater_than(sy, 0))
		{
			e.theta = atan2(m[i][j], m[i][k]);
			e.phi = atan2(sy, m[i][i]);
			e.psi = atan2(m[j][i], -m[k][i]);
		}
		else
		{
			e.theta = atan2(-m[j][k], m[j][j]);
			e.phi = atan2(sy, m[i][i]);
			e.psi = 0;
		}
	}
	else
	{
		// XYZs
		auto cy = sqrt(m[i][i]*m[i][i] + m[j][i]*m[j][i]);
		if(greater_than(cy, 0))
		{
			e.theta = atan2(m[k][j], m[k][k]);
			e.phi = atan2(-m[k][i], cy);
			e.psi = atan2(m[j][i], m[i][i]);
		}
		else
		{
			e.theta = atan2(-m[j][k], m[j][j]);
			e.phi = atan2(-m[k][i], cy);
			e.psi = 0;
		}
	}

	if(rotframe_traits<F>::odd_parity)
		e = -e;

	if(rotframe_traits<F>::rotating)
		swap(e.theta, e.psi);

	return e;
}/*}}}*/

template <rotation_frame F, class T, class A> 
Euler<F,T> to_euler(const AxisAngle<T,A> &aa)/*{{{*/
{
	return to_euler<F>(to_quaternion(aa));
}/*}}}*/

template <rotation_frame F, class T>
Euler<rotframe_traits<F>::revert,T> to_rev_euler(const Matrix<T,3,3> &m)/*{{{*/
{
	return to_euler<rotframe_traits<F>::revert>(m);
}/*}}}*/

template <rotation_frame F, class T, class A>
Euler<rotframe_traits<F>::revert,T> to_rev_euler(const AxisAngle<T,A> &aa)/*{{{*/
{
	return to_euler<rotframe_traits<F>::revert>(aa);
}/*}}}*/

// Conversion to axis-angle represention

template <class A=real, class T>
AxisAngle<T,A> to_axis_angle(const Matrix<T,3,3> &q)/*{{{*/
{
	return to_axis_angle<A>(to_quaternion(q));
}/*}}}*/

template <class A=real, rotation_frame F, class T>
AxisAngle<T,A> to_axis_angle(const Euler<F,T> &rot)/*{{{*/
{
	return to_axis_angle<A>(to_quaternion(rot));
}/*}}}*/

// UnitVector rotation

template <class T, class V, class A>
auto rotx(const V &v, A angle)/*{{{*/
	-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value) 
									&& V::dim==3, V>::type
{
    auto sa = sin(angle),
		 ca = cos(angle);
    return {v.x, v.y*ca + v.z*sa, -v.y*sa + v.z*ca};
}/*}}}*/

template <class T, class V, class A> 
auto roty(const V &v, A angle)/*{{{*/
	-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value) 
									&& V::dim==3, V>::type
{
    auto sb = sin(angle),
         cb = cos(angle);
    return {v.x*cb-v.z*sb, v.y, v.x*sb+v.z*cb};
}/*}}}*/

template <class T, class V, class A> 
auto rotz(const V &v, A angle)/*{{{*/
	-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value) 
									&& V::dim==3, V>::type
{
    auto sg = sin(angle),
         cg = cos(angle);
    return {v.x*cg+v.y*sg, -v.x*sg+v.x*cg, v.z};
}/*}}}*/

namespace detail/*{{{*/
{
	template <class T, int D, class V>
	UnitVector<T,D> conv_return(V &&v, std::identity<UnitVector<T,D>>) 
	{ 
		return unit(std::forward<V>(v)); 
	}

	template <class V, class U>
	U conv_return(V &&v, std::identity<U>) { return U(std::forward<V>(v)); }

	template <class V>
	V &&conv_return(V &&v, std::identity<V>) { return std::forward<V>(v); }
}/*}}}*/

template <class V, class T> 
auto rot(const V &v, const Matrix<T,3> &m)/*{{{*/
	-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value)
									&& V::dim==3, V>::type
{
	return detail::conv_return(m*v, std::identity<V>());
}/*}}}*/

template <class V, rotation_frame F, class T> 
auto rot(const V &v, const Euler<F,T> &e)/*{{{*/
	-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value)
									&& V::dim==3, V>::type
{
	return rot(v, to_rot_matrix(e));
}/*}}}*/

template <class T, class V, class A> 
auto rot(const V &v, const AxisAngle<T,A> &aa)/*{{{*/
	-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value)
									&& V::dim==3, V>::type
{
	return rot(v, to_quaternion(aa));
}/*}}}*/

template <template <class,int> class V, class T>
V<T,3> conv_rotframe(const V<T,3> &v, rotation_frame from, rotation_frame to)
{
	V<T,3> out;
	#if 0

	auto axis_from = axis(from),
		 axis_to = axis(to);

	out[axis_to[0]] = v[axis_from[0]];
#endif
	return out;
}

} // namespace r3

namespace r4
{

template <class A=real, class T>
AxisAngle<T,A> to_axis_angle(const UnitQuaternion<T> &q)/*{{{*/
{
	return { axis(q), angle<A>(q) };
}/*}}}*/

template <r3::rotation_frame F, class T> 
r3::Euler<F,T> to_euler(const UnitQuaternion<T> &q)/*{{{*/
{
	return r3::to_euler<F>(to_rot_matrix(q));
}/*}}}*/

template <r3::rotation_frame F, class T>
auto to_rev_euler(const UnitQuaternion<T> &q)/*{{{*/
	-> Euler<r3::rotframe_traits<F>::revert,T> 
{
	return r3::to_euler<r3::rotframe_traits<F>::revert>(q);
}/*}}}*/


template <class A, class T> 
A angle(const UnitQuaternion<T> &q)/*{{{*/
{
	A ang = 2*acos(q.w); 
	if(!equal(ang, 2*M_PI))
		return ang;
	else
		return 0;
}/*}}}*/

template <class T> 
UnitVector<T,3> axis(const UnitQuaternion<T> &q)/*{{{*/
{
	Vector<T,3> a{q.x, q.y, q.z};

	using math::sqrt;

	auto n2 = sqrnorm(a);

	if(equal(n2, 0))
		return {1,0,0};
	else
		return UnitVector<T,3>{a / sqrt(n2), true};
}/*}}}*/

template <class T, class V> 
auto rot(const V &v, const UnitQuaternion<T> &q)/*{{{*/
	-> typename std::enable_if<(is_vector<V>::value || is_point<V>::value)
									&& V::dim==3, V>::type
{
	auto q1 = q*Quaternion<T>(v);
	auto q2 = q1*inv(q);

	return r3::detail::conv_return(imag(q2), std::identity<V>());
}/*}}}*/

template <class T>
Matrix<T,3,3> to_rot_matrix(const UnitQuaternion<T> &q)/*{{{*/
{
	T xs = q.x*2,  ys = q.y*2,  zs = q.z*2,
	  wx = q.w*xs, wy = q.w*ys, wz = q.w*zs,
	  xx = q.x*xs, xy = q.x*ys, xz = q.x*zs,
	  yy = q.y*ys, yz = q.y*zs, zz = q.z*zs;

	return Matrix<T,3,3>
	{
		{ 1-(yy + zz),   xy - wz,     xz + wy  },
		{    xy + wz, 1-(xx + zz),    yz - wx  },
		{    xz - wy,    yz + wx,  1-(xx + yy) }
	};
}/*}}}*/


}}} // namespace s3d::math::r4
