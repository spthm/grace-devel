#ifndef S3D_UTIL_RAND_H
#define S3D_UTIL_RAND_H

#include <stdlib.h>
#include <functional>

namespace s3d 
{

inline double rand(double min, double max)
{
	return ::rand()/double(RAND_MAX)*(max-min)+min;
}

template <class T> void srand(const T &seed)
{
	::srand(std::hash<T>()(seed));
}

}

#endif
