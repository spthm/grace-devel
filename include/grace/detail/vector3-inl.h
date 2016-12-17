#pragma once

// No grace/vector.h include.
// This should only ever be included by grace/vector.h.
#include <grace/error.h>

namespace grace {

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>& Vector<3, T>::operator=(const Vector<3, U>& rhs)
{
    Base::operator=(rhs);
    return *this;
}

template <typename T>
GRACE_HOST_DEVICE
T& Vector<3, T>::operator[](int i)
{
    switch (i) {
        case 0: return this->x;
        case 1: return this->y;
        case 2: return this->z;
    }
    GRACE_ASSERT(0, vector3_invalid_index_access);
}

template <typename T>
GRACE_HOST_DEVICE
const T& Vector<3, T>::operator[](int i) const
{
    switch (i) {
        case 0: return this->x;
        case 1: return this->y;
        case 2: return this->z;
    }
    GRACE_ASSERT(0, vector3_invalid_index_access);
}

} // namespace grace
