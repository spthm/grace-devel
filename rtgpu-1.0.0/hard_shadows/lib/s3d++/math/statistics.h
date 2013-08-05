#ifndef S3D_MATH_STATISTICS_H
#define S3D_MATH_STATISTICS_H

#include <vector>

namespace s3d { namespace math
{

template <class T, int D>
Matrix<T,D,D> covariance(const std::vector<Vector<T,D>> &samples);

template <class T>
T variance(const std::vector<T> &samples);

template <class T>
T mean(const std::vector<T> &samples);

}}

#include "statistics.hpp"

#endif
