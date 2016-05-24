#pragma once

namespace grace {

template<typename T>
struct RayData_datum
{
    T data;
};

template <typename T, typename Real>
struct RayData_sphere
{
    T data;
    Real b2, dist;
};

} // namespace grace
