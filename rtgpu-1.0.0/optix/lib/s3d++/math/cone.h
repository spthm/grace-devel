#ifndef S3D_MATH_CONE_H
#define S3D_MATH_CONE_H

#include "fwd.h"

namespace s3d { namespace math
{

template <class T, int D>
struct Cone
{
	Cone() {}
	Cone(const Point<T,D> &o, const Vector<T,D> &d, T _cosa)
		: origin(o), dir(d), cosa(_cosa) {}

	Point<T,D> origin;
	Vector<T,D> dir;
	T cosa;

	bool is_coupled_to(const Vector<T,D> &v) const;

private:
#if HAS_SERIALIZATION
	friend class boost::serialization::access;

	template <class A>
	void serialize(A &ar, unsigned int version)
	{
		ar & origin;
		ar & dir;
		ar & cosa;
	}
#endif
};

template <class T, int D>
bool are_coupled(const Cone<T,D> &c1, const Cone<T,D> &c2);

}} 

#include "cone.hpp"

#endif
