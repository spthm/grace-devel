namespace s3d { namespace math { namespace r3
{

template <class T>
T fov_horiz(const Frustum<T> &f)
{
	return f.fov;
}

template <class T>
T fov_vert(const Frustum<T> &f)
{
	return atan(tan(f.fov/2)/f.aspect)*2;
}

template <class T>
Plane<T,3> top_plane(const Frustum<T> &f)
{
	auto right = unit(cross(f.axis, f.up));
	return { f.pos, rot(-f.up, r3::axis_angle{right, fov_vert(f)/2}) };
}

template <class T>
Plane<T,3> bottom_plane(const Frustum<T> &f)
{
	auto right = unit(cross(f.axis, f.up));
	return { f.pos, rot(f.up, r3::axis_angle{right, -fov_vert(f)/2}) };
}

template <class T>
Plane<T,3> near_plane(const Frustum<T> &f)
{
	return { f.pos + f.axis*f.near, f.axis };
}

template <class T>
Plane<T,3> far_plane(const Frustum<T> &f)
{
	return { f.pos + f.axis*f.far, -f.axis };
}

template <class T>
Plane<T,3> left_plane(const Frustum<T> &f)
{
	auto right = unit(cross(f.axis, f.up));
	return { f.pos, rot(right, r3::axis_angle{f.up, fov_horiz(f)/2}) };
}

template <class T>
Plane<T,3> right_plane(const Frustum<T> &f)
{
	auto right = unit(cross(f.axis, f.up));
	return { f.pos, rot(-right, r3::axis_angle{f.up, fov_horiz(f)/2}) };
}

template <class T, class R>
Frustum<T> &rot_inplace(Frustum<T> &f, const R &r)
{
	rot(f.axis, r);
	rot(f.up, r);
	return f;
}

template <class T, class R>
Frustum<T> &&rot(Frustum<T> &&f, const R &r)
{
	return std::move(rot_inplace(f, r));
}

template <class T, class R>
Frustum<T> rot(const Frustum<T> &f, const R &r)
{
	return rot(Frustum<T>(f));
}


}}} // namespace s3d::math::r3
